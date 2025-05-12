---
title: "Week 17"
date: 2025-04-27
description: "Writing Modular Arithmetic in MLIR."
keywords: ["mlir", "modular", "arithmetic", "tech", "blog"]
draft: false
tags: ["mlir"]
summary: Writing Modular Arithmetic in MLIR.
---

This blog post presents two ways to write Modular Arithmetic in MLIR. 
The sentences below sometimes make statements that are not necessarily correct yet, and refinement is to follow. 
This writing serves primarily to consolidate my second week of grappling with the compiler framework. 

# 1. Writing Modular Arithmetic with a Custom IR.

Let’s write some MLIR code to perform a modular addition.
With MLIR, we can define custom Intermediate Representation that succinctly describe what operations we want to perform. It naturally makes sense to write an IR for modular arithmetic. 

We start by defining an IR for modular arithmetic by creating our own type for prime field elements. 
We’ll fix our type to elements in the Baby Bear prime field (`p = 2**31-2**27+1`) for now, rather than making the modulus generic. 
The Baby Bear prime is a 31-bit element, and fits comfortably in the built-in MLIR `IntegerType(32)` type, which is conveniently aliased to `i32`. 
We could type alias the existing `i32` to our Baby Bear type as `bb31 = IntegerType(32)`, but we'll just work with `i32` for now. 

Knowing our types, we can now define our modular addition operation through its inputs and output, or in MLIR terminology, its arguments and result. 
Taking inspiration from the `ModArith` IR in [Google’s HEIR project](https://heir.dev/docs/dialects/modarith/), we can define our own modular addition as `mod_arith_bb31.add`. 
With the xDSL framework, this operation is defined as follows. 
```python
@irdl_op_definition
class ModAddOp(IRDLOperation):
    name = "mod_arith_bb31.add"

    # No idea what traits are yet...
    traits = traits_def(Pure())

    # Define the operands and the result
    lhs = operand_def(i32)
    rhs = operand_def(i32)
    result = result_def(i32)

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(operands=[lhs, rhs], result_types=[i32])
```

To use this operations, we can wrap it in a function, which we then have to wrap in a module. 
We can write this by hand in MLIR, but we'll use xDSL's `Builder` for simplicity. 
```python
# Create a module, and create a builder at its first block
module = ModuleOp([])
builder = Builder(InsertPoint.at_end(module.body.block))

# Create the MLIR types for the function
in_arg_types = [i32, i32]
res_types = [i32]

# Create the function and insert it inside the module
func = FuncOp("main", (in_arg_types, res_types))
builder.insert(func)

# Set the builder insertion point inside the function and create a modular addition
builder.insertion_point = InsertPoint.at_end(func.body.block)
result = ModAddOp.build(
	    operands=[func.args[0], func.args[1]], 
	    result_types=func.function_type.outputs
	)
builder.insert(result)

# Create a return statement
builder.insert(ReturnOp(result))
```

The `print(module)` command then gives the following good-looking and readable MLIR code.
```mlir
builtin.module {
  func.func @main(%0 : i32, %1 : i32) -> i32 {
    %2 = "mod_arith_bb31.add"(%0, %1) : (i32, i32) -> i32
    func.return %2 : i32
  }
}
```

In terms of execution, there is little usefulness here. We can't compile and run anything, because we only defined a syntax! 
We’ll at least have to rewrite this functionality in a different way, and then handle a way to convert one to the other. While I don’t know if this is true yet, I assume there is some benefit in lowering this operation to MLIR’s stock IRs, and particularly to the `arith` dialect for defining standard arithmetic operations. 

# 2. Writing Modular Arithmetic with the Built-in `arith` IR.
With xDSL's `Builder`, we fill out the functionality of the modular addition operation in terms of basic arithmetic and control flow as follows.
```python
# Create a module, and create a builder at its first block
module_arith = ModuleOp([])
builder = Builder(InsertPoint.at_end(module_arith.body.block))

# Create the MLIR types for the function
in_arg_types = [i32, i32]
res_types = [i32]

# Create the function and insert it inside the module
func = FuncOp("main", (in_arg_types, res_types))
builder.insert(func)

# Set the builder insertion point inside the function
builder.insertion_point = InsertPoint.at_end(func.body.block)

# Create the possible output values of the operation
modulo = builder.insert(
             ConstantOp(IntegerAttr.from_int_and_width(value=2013265921, width=32))
         )
res_no_overflow = builder.insert(AddiOp(func.args[0], func.args[1]))
res_overflow = builder.insert(SubiOp(res_no_overflow, modulo))

# Create the selector to choose the right output value 
is_lt = builder.insert(CmpiOp(res_no_overflow, modulo, "ult"))
res_select = builder.insert(SelectOp(is_lt, res_no_overflow, res_overflow))

# Create a return statement
builder.insert(ReturnOp(res_select))
```

With `print(module_arith)`, this prints to still readable but definitely more involved, MLIR code.
```mlir
builtin.module {
  func.func @main(%0 : i32, %1 : i32) -> i32 {
    %2 = arith.constant 2013265921 : i32
    %3 = arith.addi %0, %1 : i32
    %4 = arith.subi %3, %2 : i32
    %5 = arith.cmpi ult, %3, %2 : i32
    %6 = arith.select %5, %3, %4 : i32
    func.return %6 : i32
  }
}
```
The contrast between the two resulting code snippets exposes a big advantage of MLIR: 
our custom modulo-aware IR is nicer to read when performing calculations in the modular domain, whereas the built-in `arith` IR is nicer to interpret by a standard CPU, that do not natively support modular arithmetic. 
This domain specification is what allows optimizations to be expressed in the language that makes the most sense for the optimization. This enhances readability, maintainability, and leads to more optimal performance at the end of the compiler chain. 


# Open Loops
As the island of knowledge grows, so do the shores of our ignorance, and we're left with some open questions.
 - [X] [How do we generically specify our own modular arithmetic type such that we can configure the modulus?]({{< relref "mlir-01.md" >}})
 - [ ] What are traits in operations?
 - [ ] How do Modules, Blocks, Functions, and Operations relate? What other structures are there? 
 - [ ] How do we lower one module to another? 
 - [X] [How do we run our modules?]({{< relref "mlir-02.md" >}})

# Next-Up
The first topic we'll be diving into is the generalization of our modulo operation by defining a type where we can set the modulus. 
That way, we avoid having to define a new dialect for each prime field. 
