---
title: "Lowering One Dialect to Another"
date: 2025-05-24
description: "Lower One Dialect to Another."
keywords: ["mlir", "modular", "arithmetic", "tech", "blog"]
draft: false
tags: ["mlir"]
summary: Week 21, San Diego --- Closing the compiler loop.
---

In the past weeks, we've created an IR for modular arithmetic through an MLIR dialect, and have tested the modular arithmetic operations expressed in the `arith` dialect using an interpreter. 
The following two snippets should be a quick recap away.
```mlir
builtin.module {
  func.func @main(%0 : !mod_arith.int<2013265921 : i32>, 
  	%1 : !mod_arith.int<2013265921 : i32>) -> !mod_arith.int<2013265921 : i32> {
    %2 = "mod_arith.add"(%0, %1) : (!mod_arith.int<2013265921 : i32>, 
    	!mod_arith.int<2013265921 : i32>) -> !mod_arith.int<2013265921 : i32>
    func.return %2 : !mod_arith.int<2013265921 : i32>
  }
}
```

```mlir
builtin.module {
  func.func @main(%0 : i32, %1 : i32) -> i32 {
    %2 = arith.addi %0, %1 : i32
    %3 = arith.constant 2013265921 : i32
    %4 = arith.cmpi ult, %2, %3 : i32
    %5 = arith.subi %2, %3 : i32
    %6 = arith.select %4, %2, %5 : i32
    func.return %6 : i32
  }
}
```
We have tested the bottom module, and can assume here that `arith` is our 'exit' dialect, as it 'runs' somewhere (which is at the moment in an interpreter).
We do want to avoid writing the ops in the bottom function everytime we want to perform a modular addition. In that case, we'd prefer to use the more concise and readable top module; which is actually the reason why we've written the `ModArith` dialect.
To get the best of both worlds, we'll write a compiler pass, or lowering, to convert the `ModArith` dialect to the `arith` dialect. This allows us to both write modules easily and execute them on a platform. 

I wrapped the operations in a `func.func`, and initiated the dialect conversion from there. This was less trivial than just converting operations and gave me some headaches; meaning there might be some value in documenting this. The latter is covered very well in [xDSL's Defining Dialects tutorial](https://xdsl.readthedocs.io/stable/marimo/defining_dialects/). 

Our regular disclaimer applies: what's written here is not necessarily entirely correct, and I even suspect that the approach that follows is not entirely the right way of performing a dialect conversion.


## 1. Lowering the Operations.
The easiest part of the pass is to scan the operations in our function for `ModAddOp` operations, and replace them with a sequence of operations from the `arith` dialect that implement the right functionality.
The formula for modular addition is `mod_add(x, y) = (x + y)` when `(x + y) < modulo` and `mod_add(x, y) = (x + y) - modulo` otherwise. We use `arith.select` for control flow rather than the (more appropriate?) structured control flow dialect (`scf`) for simplicity.

```python3
class LowerAddOp(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        # Implement the lowering of ModAddOp
        if not isinstance(op, ModAddOp):
            return

        no_overflow = rewriter.insert(AddiOp(op.lhs, op.rhs)).result
        modulo = rewriter.insert(ConstantOp(IntegerAttr.from_int_and_width(
        	value=op.result.type.modulus.value, width=32))).result
        is_lt = rewriter.insert(CmpiOp(no_overflow, modulo, "ult")).result
        overflow = rewriter.insert(SubiOp(no_overflow, modulo)).result
        select = rewriter.insert(SelectOp(is_lt, no_overflow, overflow)).result
        rewriter.replace_matched_op([], new_results=[select])
        return select
```


## 2. Converting Function Argument Types.
At this point, our MLIR code is invalid; we did not specify a cast operation from the `mod_arith.int` type to the `i32` type. 
We either have to rectify this during the previous pass, or fix this in post, i.e. in later passes. Since we won't be verifying the validity of our MLIR code between passes, but only after a set of passes are applied, we will create other passes to address the casting. 
A function that derives from `TypeConversionPattern` is used for the type conversion we need. 
This unfortunately does not convert the return types of the function signature, and we'll need to define an additional pass. 

```python3
class ModToArithTypeConversion(TypeConversionPattern):
    @attr_type_rewrite_pattern
    def convert_type(self, typ: ModArithType) -> i32:
        return i32
```

## 3. Convert the Function Return Types.
We have one last modification to pull through. The return type of our function signature needs conversion to `i32`. This is a bit more verbose---and cumbersome---than our earlier passes. 
The main idea is to first check whether the signature's return types need conversion, and return if they don't. If not, this somehow falls into an endless loop where this pass just keeps getting called and nothing is converted. I'll have to dive deeper into pattern rewriting to find out why though. 
When the signature's return types do need conversion, we create a new function type that keeps the previously-updated input types, while assigning our desired `i32` output type. 
We can then tie this pair of new types to a new function, which takes over the same name, and same content of the function we want to convert the signature of. 
The tricky part here is that we need to move the original function body (which is---as you undoubtedly recall from last week---a region) over as well. 
Once we created the new function, we replace the old one. Out with the old in with the new.

```python3
class ConvertFunctionReturnTypePattern(RewritePattern):
    """Converts function return types from ModArithType to i32."""
    
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        # Check if this is a function operation
        if not isinstance(op, FuncOp):
            return
            
        # Get the current function type
        func_type = op.function_type
        
        # Check if any output types need conversion
        needs_conversion = False
        for output_type in func_type.outputs:
            if str(output_type).startswith("!mod_arith.int"):
                needs_conversion = True
                break
                
        if not needs_conversion:
            return
        
        # Create a new function type with the same inputs but converted outputs
        new_func_type = FunctionType.from_lists(
            # Keep the previously-converted inputs
            inputs=func_type.inputs,
            outputs=[i32]
        )
        
        # Create a new function with the updated type
        new_func = FuncOp(
            op.sym_name.data,
            new_func_type,
            rewriter.move_region_contents_to_new_regions(op.body)
        )
        
        # Replace the old function with the new one
        rewriter.replace_matched_op(new_func) 
```

## 4. Rewriting the Module.
Now we can convert the individual ingredients of our `mod_add` module, we can combine them in a list of 'rewrite patterns'. 
The rewrite patterns are passed to a `PatternRewriteWalker`, which applies the functions in this list one by one to transform the module accordingly. 
Here, I took over the `GreedyRewritePatternApplier` from code example, so there might be alternatives that fit better for this case. That's to be investigated though. 
Applying this `lower_mod_arith` to our `mod_add` operation gives us the module we tested two weeks ago with the interpreter. In other words, our first lowering pass works!

```python3
# Lower the mod_arith dialect
def lower_mod_arith(op: Operation):
    # Add rewrite patterns in this list
    rewrites = [
        LowerAddOp(),
        ModToArithTypeConversion(),
        ConvertFunctionReturnTypePattern(),
    ]

    PatternRewriteWalker(GreedyRewritePatternApplier(rewrites)).rewrite_module(
        op
    )
```


# Next-Up
We just accomplished a full compiler loop. We now have a working pass between the code snippets written in the `ModArith` and `Arith` dialects we saw in our first week. 
When I set out documenting my struggles with MLIR 5 weeks ago, I started sharing my experiences during my second week, skipping the first. 
What we've just done is what I attempted---and failed---to do in my first week. 
Getting the basics took me a lot longer than expected. I'm happy I made it this far. 

We've learned a lot on the way, and it's time to capitalize on our learning. 
We'll shelve our (unwieldy) stack of open questions for a week, and look at an application of our `ModArith` dialect. 
That should give us an idea of what's possible with this beautiful technology.
MLIR is a very powerful compiler framework for performance; the [Mojo programming language](https://www.modular.com/mojo) for instance is close to 5 orders of magnitude faster than Python thanks to MLIR. My expectations (and hopes) are high!

