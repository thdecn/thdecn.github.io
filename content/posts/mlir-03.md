---
title: "Traversing MLIR Code"
date: 2025-05-17
description: "Traversing MLIR Code."
keywords: ["mlir", "modular", "arithmetic", "tech", "blog"]
draft: false
tags: ["mlir"]
summary: Week 20, San Francisco --- Reading the docs for some theory, Part 1.
---

This week, we have a more theory-heavy post. 
We will attempt to make sense of Operations, Blocks, and Regions, as well as Types, Properties, and Attributes. 
To make it slightly easier to digest, I added a picture this time. 
Keeping it more visual, and hopefully more intuitive should help us create the beginnings of an xDSL/MLIR cheat sheet we can rely on later in more ambitious projects. 

The content here is rephrased from the MLIR docs; 
the [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/) and 
the [Understanding the IR Structure](https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/) tutorial. 
We follow the latter closely.

This article is really a mental model of my current understanding of MLIR's language and IR structure. None of this is necessarily correct. And over time, can be updated and refined as my understanding evolves.
 
## 1. Walking the IR.
To start, we'll need a module. We've seen module creation using the `Builder` and `ImplicitBuilder` in the past. 
We'll add a third approach to our toolkit by writing an MLIR code snippet as a text file; before parsing it into a module we can work with. The code below looks different from the MLIR we've seen before as it is written more formally. 

```python3
mlir_text = """\
"builtin.module"() ( {
  %results:4 = "dialect.op1"() {"attribute name" = 42 : i32} : 
  	() -> (i1, i16, i32, i64)
  "dialect.op2"() ( {
    "dialect.innerop1"(%results#0, %results#1) : (i1, i16) -> ()
  },  {
    "dialect.innerop2"() : () -> ()
    "dialect.innerop3"(%results#0, %results#2, %results#3)[^bb1, ^bb2] : 
    	(i1, i32, i64) -> ()
  ^bb1(%1: i32):  // pred: ^bb0
    "dialect.innerop4"() : () -> ()
    "dialect.innerop5"() : () -> ()
  ^bb2(%2: i64):  // pred: ^bb0
    "dialect.innerop6"() : () -> ()
    "dialect.innerop7"() : () -> ()
  }) {"other attribute" = 42 : i64} : () -> ()
}) : () -> ()\
"""
```
Before invoking the parser on this text, we set up a compiler context and register the dialects we use as well as allow dialects that are not defined (i.e. are unregistered).
```python3
ctx = Context()
ctx.allow_unregistered = True
ctx.load_dialect(builtin.Builtin)

mlir_module = Parser(ctx, mlir_text).parse_module()
```
After parsing, `mlir_module` holds our module.

Code compilation is a sequence of passes that optimize and lower the operations until they are translated to a list of instructions that are supported by the target platform we're compiling to. 
Every pass will have to analyze certain operations and eventually transform them. 
The analysis is done by walking the IR, which we can do from top to bottom using the iterator that is exposed by the `walk()` function of the `Operation` class. 

We can get the names of all the operations in our module as follows.
```python3
print([op.name for op in mlir_module.walk()])
```
Great, that's something. We can now sequentially go through the module and list all operations.
This could help us count operations, for example, to estimate performance. 

## 2. Traversing the IR Nesting.
The [Understanding the IR Structure](https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/) tutorial is a great way to test some MLIR to xDSL conversion.
We are given the example IR we used above, three methods that are used to recursively print its properties, as well as the resulting output. 
Using these three elements, we can Rosetta Stone our way from MLIR C++ to Python xDSL. 

Let's start from the top; as the module itself is an operation (`builtin.module`), we start with a `printOperation()` function. The IR is recursively nested, and we recurse into the Regions attached to the Operation. 

```python3
def printOperation(op: builtin.Operation, indent: Indent) -> str:
    # Print the operation and some of its properties
    if op.name == "builtin.unregistered":
        op_name = op.__str__()
        # The unregistered operation's name is between quotation marks
        start = op_name.find("\"") + 1
        end = op_name.find("\"", start)
        op_name = op_name[start:end]
    else:
        op_name = op.name

    op_str = indent.printIndent() + "visiting op: '" + op_name + "' with " 
    	+ f"{len(op.operands)}" + " operands and " 
    	+ f"{len(op.results)}" + " results\n"

    # Print the operation attributes
    if len(op.attributes) > 1:
        op_str += indent.printIndent() 
        	+ f"{len([key for key in op.attributes])-1}" + " attributes:\n"
        
        for key, value in op.attributes.items():
            if key != "op_name__":
                op_str += indent.printIndent() 
                	+ " - '" + key + "' : '" + f"{value}" + "'\n"

    # Recurse into each of the regions attached to the operation.
    op_str += indent.printIndent() + " " 
    	+ f"{len(op.regions)}" + " nested regions:\n"
    indent.pushIndent()
    for region in op.regions:
        indent_rec = Indent(indent.level)
        op_str += printRegion(region, indent_rec)
    return op_str
```
We print the Regions, and as each Region is a list of Blocks, we recurse into them next.
```python3
def printRegion(region: builtin.Region, indent: Indent) -> str:
    # A region does not hold anything other than a list of blocks.
    reg_str = indent.printIndent() 
    	+ "Region with " + f"{len(region.blocks)}" + " blocks:\n"
    indent.pushIndent()
    for block in region.blocks:
        indent_rec = Indent(indent.level)
        reg_str += printBlock(block, indent_rec)
    return reg_str
```

Lastly, we print the properties of each Block, and recurse into its list of Operations. 
```python3
def printBlock(block: builtin.Block, indent: Indent) -> str:
    # Print the block intrinsics properties (essentially the argument list)
    block_str = indent.printIndent() 
    	+ "Block with " + f"{len(block.args)}" + " arguments, " 
    	+ f"{len(list(block.last_op.successors))}" + " successors, and "
    	+ f"{len(list(block.ops))}" + " operations\n"

    # A block's main role is to hold a list of Operations: 
    #  we recurse into printing each operation.
    indent.pushIndent()
    for op in block.ops:
        block_str += printOperation(op, indent)
    return block_str
```
This prints exactly what we expect from recreating the tutorial. 
```
visiting op: 'builtin.module' with 0 operands and 0 results
 1 nested regions:
  Region with 1 blocks:
    Block with 0 arguments, 0 successors, and 2 operations
      visiting op: 'dialect.op1' with 0 operands and 4 results
      1 attributes:
       - 'attribute name' : '42 : i32'
       0 nested regions:
      visiting op: 'dialect.op2' with 0 operands and 0 results
        1 attributes:
         - 'other attribute' : '42 : i64'
         2 nested regions:
          Region with 1 blocks:
            Block with 0 arguments, 0 successors, and 1 operations
              visiting op: 'dialect.innerop1' with 2 operands and 0 results
               0 nested regions:
          Region with 3 blocks:
            Block with 0 arguments, 2 successors, and 2 operations
              visiting op: 'dialect.innerop2' with 0 operands and 0 results
               0 nested regions:
              visiting op: 'dialect.innerop3' with 3 operands and 0 results
               0 nested regions:
            Block with 1 arguments, 0 successors, and 2 operations
              visiting op: 'dialect.innerop4' with 0 operands and 0 results
               0 nested regions:
              visiting op: 'dialect.innerop5' with 0 operands and 0 results
               0 nested regions:
            Block with 1 arguments, 0 successors, and 2 operations
              visiting op: 'dialect.innerop6' with 0 operands and 0 results
               0 nested regions:
              visiting op: 'dialect.innerop7' with 0 operands and 0 results
               0 nested regions:
```

We have delved a bit deeper into the definitions of `Operation`, `Region`, and `Block`, which we can summarize into the picture below, where operations are depicted in white, regions in purple, and blocks in green. 

{{< figure src="/images/MLIR-IR-Structure.png" title="MLIR's Recursive IR Structure" >}}


## 3. Defining Terminology.
 - **Operations.** Operations are the atomic building blocks of an IR. They are identified by a unique name, return zero or more results, and take in zero or more operands or arguments. They optionally contain a dictionary of attributes and/or a dictionary of properties. 
They have zero or more successors of type `BlockArgument`, and hold a list of zero or more regions of type `Region`. 
Examples of operations are the integer addition operation `arith.addi` and the top-level container operation `builtin.module`.

 - **Blocks.** A Block, or a compiler basic block, is a list of operations that are executed in order and terminated by a terminator operation. In other words, there is no control flow within a block. Blocks can take in inputs, which are referred to as `BlockArguments`. When these are arguments of a region's entry block, they are also arguments to that region. The block arguments of other blocks are determined by terminator operations that have the block as a successor.

 - **Region.** A region is an ordered list of Blocks. Operations can have any number of (nested) regions. The first block in the region is the entry block, and its arguments are the region's arguments. No other block can list the entry block as a successor. An example of a region is a function body; it forms a control flow graph of blocks, where block terminators branch either to different blocks or return the function.
The return types match the number of results and the types of the function's signature, and the function's arguments match the number and types of the region's arguments
Regions can essentially be used to open a new scope.no type, attributes, or properties. 
Regions themselves have no type, no attributes, and no properties. 

 - **SSA Values.** SSA values are defined as either results (when they are outputs) or block arguments (when they are inputs). Each value has a specific type, which is defined and used by operations. SSA stands for Static Single Assignment, which is a form where values can only be assigned once. 

 - **Types.** Each MLIR value has a type defined by the type system at compile time. Types can be aliased to be referable by a different, more concise name. Similar to operations, dialects can define custom types to extend the type system without limitation. The `builtin` dialect provides a set of types that are directly usable by any other dialect. 

 - **Attributes.** Attributes are used to tie constant---rather than variable---data to operations at compile time. Attributes are stored in a dictionary, associating attribute names or keys to attribute values. The `builtin` dialect provides a rich set of attribute values that are directly usable by any other dialect, such as arrays, dictionaries, strings, integers etc. 
Attributes are used for optional metadata that another tool might add or remove; they are discardable.

 - **Properties.** Properties are attributes inherent to the definition of an operation, as well as any other arbitrary data that can be tied to operations. The data can be accessed through interface accessors and other methods. 
Properties are used when a value is part of the operation’s definition and verifier.

A minimal example explains the difference between an attribute and the more recently introduced property.
```C++
// Old style: predicate is an ATTRIBUTE
%cmp = arith.cmpi %a, %b {predicate = #arith.cmpi<"slt">} : i32

// After the arith dialect adopted PROPERTIES
%cmp = arith.cmpi slt, %a, %b : i32
// The 'slt' token is backed by a field in cmpi's properties struct;
// nothing appears in the attribute dictionary of the printed IR.
```

## 4. Cheat Sheeting.
During our translation exercises, we collected a table to go from MLIR C++ to xDSL Python. 
This can help in the future, when we port MLIR to xDSL as a basis for further experimentation.
| MLIR C++                    | xDSL Python             |
| --------                    | -------                 |
| `op->getName()`             | `op.name`               |
| `op->getRegions()`          | `op.regions`            |
| `op->getNumRegions()`       | `len(op.regions)`       |
| `region.getBlocks()`        | `region.blocks`         |
| `region.getBlocks().size()` | `len(region.blocks)`    |
| `block.getOperations()`     | `block.ops`             |
| `block.getOperations().size()`  | `len(list(block.ops))` |
| `op->getNumOperands()`      | `len(op.operands)`      |
| `block.getNumArguments()`   | `len(block.args)`       |
| `block.getNumSuccessors()`  | `len(block.successors)` |
| `op->getNumResults()`       | `len(op.results)`       |
| `op->getAttrs().size()`     | `len(op.attributes)`    |
| `op->getAttrs().empty()`    | `len(op.attributes)!=0` |


# Open Loops
The greater our understanding, the broader the horizon of the unknown. 
We amassed a good heap of open questions, and are adding a few more this week. 
They are now clustered to keep things organized. 

 - **Practical topics.**
 - [X] [How do we lower one module to another?]({{< relref "mlir-04.md" >}})
 - [ ] <mark>**NEW!**</mark> How do we estimate the performance of a module?
<br><br>
 - **Small changes.**
 - [ ] How do we set default values and default types? 
 - [ ] Our `mod_arith.int` type is a bit verbose, can we print a string rather than a value to clarify the prime? 
 - [ ] How do we constraint the signedness and canonicalization of the inputs? 
 - [ ] How do we catch overflows in the `arith` dialect?
<br><br>
 - **Theory.**
 - [ ] What are traits in operations used for?
 - [ ] Does a type need an associated attribute like `IntegerType` has `IntegerAttr`?
 - [ ] How does xDSL/MLIR structure code? What can I learn from going through e.g. the `builtin` dialect?
 - [ ] What other ways are there to test operations and functions in a dialect? Is there a way that does not use an interpreter? 
 - [ ] What are the different ways to create MLIR code? How does the `Builder` compare to the `ImplicitBuilder`? 
 - [ ] <mark>**NEW!**</mark> What are some interesting xDSL/MLIR projects? How useful is CIRCT for hardware? Is IREE useful for runtimes?
 - [ ] <mark>**NEW!**</mark> What is the elementary subset of compiler terminology to be productive? What is e.g. Dead Code Elimination (DCE), Common Subexpression Elimination (CSE), and Loop-Invariant Code Motion (LICME). What is a Definition-Use (DU) Chain? What are PHI nodes, what are alternatives to SSA representation? 

# Next-Up
Oof, we survived a difficult one. 
The writing here unfortunately did end up being a bit of a brick wall. 
Time to get back to our original goal: converting (or lowering) the operations in our modular arithmetic dialect (`mod_arith`) into native MLIR dialects (e.g. `builtin`, `func`, `arith`, ...). 
Once there, we can easily write algorithms featuring modular arithmetic and compile them down to an IR we can evaluate using the interpreter.
Hopefully, when we tackle this lowering next week, we'll benefit at least a bit from the heavy lifting we did here. We'll see.

