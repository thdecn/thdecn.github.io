---
title: "Evaluating Optimization Passes"
date: 2025-06-08
description: "Evaluating Optimization Passes."
math: true
keywords: ["mlir", "modular", "arithmetic", "poseidon", "hashing", "zk", "tech", "blog"]
draft: false
tags: ["mlir", "zk"]
summary: Week 23 --- First passes to optimize our Poseidon hash.
---

Last week, we coded a toy Poseidon2 hash function, lowered it to the `arith` dialect, and evaluated it based on the frequency of the different `arith` operations. 
This week, we will attempt to significantly reduce the frequencies associated with each op by studying common mid-end compiler passes, and applying them to our code. 
The `arith` operations, for now, form our target ISA, which is why we use it to bench our compiler optimizations. Once we start considering the compiler back-end, a realistic ISA will take its place. We do want to get into the habit of acting with performance in mind, and a short feedback loop does help 

More briefly, our goal this week is to describe and apply an subset of elementary mid-end compiler passes to our Poseidon2 hash, and measure the overall impact on the operation (think instruction) count. 

The idea to use xDSL over C++ MLIR was that, aside from not having to be bogged down with going pro with C++[^1] and CMake, I could just use Python[^2], a language I can read and debug in productively. 
This allows me to not only learn MLIR faster, it allows me to rely on [Cursor](https://www.cursor.com/) to help me generate boilerplate as well as (more or less) functional code. 
Having it code-up random queries would not help my learning of course, but having it assist in writing and explaining compiler optimization passes for instance definitely accelerates it. 
The majority of the passes here were written by individual queries to Cursor (followed by some minor requerying and debugging). 
The learnings from them will be synthesized next week in an 'Anatomy of a Pass' section. 

The drawback is of course that it is easy to come up with alternative, exo-MLIR passes that fit in the framework, but that might already be built in (MLIR comes with its own batteries). 
A lot of the passes that follow are likely better to implement using traits or interfaces, which we have yet looked into. 
We would not be using MLIR to the fullest of its potential by leaving traits by the wayside. 
This would also harm our learning, and we'll therefore have to make sure we get this information from a meaty expert. 

So rather than overloading you with AI-generated (but useful) slop this week, this blog presents you some human-written-but-not-edited wall-of-text slop instead; proceed at your own peril!
It might be better to learn about mid-end compilers somewhere else for now. 
Let me know if you find a better resource though, I’m still up for editing this one based on it.

[^1]: Don't get me wrong, I love C! I'm just hopelessly confused with C++ and all its versions.
[^2]: As far as I'm concerned, every engineer is a python native. Also, yes, I just learned how to write footnotes in Markdown. 

# 1. First Passes
<mark>**Loop Unrolling.**</mark> Loop Unrolling is a known compiler optimization that increases the performance by reducing the loop control instructions at the cost of increasing the code size. 
Down the line, it also allows for better scheduling in multiple-issue processors (e.g. VLIW or superscalar processors). 
We don't expect a huge performance increase from this pass, but start here to illustrate an optimization pass. 
We write an `UnrollForLoopPattern(RewritePattern)` and apply it with the `PatternRewriteWalker` giving us the following code. 
```mlir
func.func @main(%0 : bb31, %1 : bb31, %2 : bb31, %3 : bb31) -> (bb31, bb31, bb31, bb31) {
    %4 = arith.constant 0 : i32
    %5 = arith.constant 4 : i32
    %6 = arith.constant 1 : i32
    %7 = arith.constant 0 : i32
    %8, %9, %10, %11 = func.call @ArcRf(%0, %1, %2, %3) : (bb31, bb31, bb31, bb31) -> (bb31, bb31, bb31, bb31)
    %12, %13, %14, %15 = func.call @SboxRf(%8, %9, %10, %11) : (bb31, bb31, bb31, bb31) -> (bb31, bb31, bb31, bb31)
    %16, %17, %18, %19 = func.call @MatMulRf(%12, %13, %14, %15) : (bb31, bb31, bb31, bb31) -> (bb31, bb31, bb31, bb31)
    %20 = arith.constant 1 : i32
    %21, %22, %23, %24 = func.call @ArcRf(%16, %17, %18, %19) : (bb31, bb31, bb31, bb31) -> (bb31, bb31, bb31, bb31)
    %25, %26, %27, %28 = func.call @SboxRf(%21, %22, %23, %24) : (bb31, bb31, bb31, bb31) -> (bb31, bb31, bb31, bb31)
    %29, %30, %31, %32 = func.call @MatMulRf(%25, %26, %27, %28) : (bb31, bb31, bb31, bb31) -> (bb31, bb31, bb31, bb31)
    %33 = arith.constant 2 : i32
    %34, %35, %36, %37 = func.call @ArcRf(%29, %30, %31, %32) : (bb31, bb31, bb31, bb31) -> (bb31, bb31, bb31, bb31)
    %38, %39, %40, %41 = func.call @SboxRf(%34, %35, %36, %37) : (bb31, bb31, bb31, bb31) -> (bb31, bb31, bb31, bb31)
    %42, %43, %44, %45 = func.call @MatMulRf(%38, %39, %40, %41) : (bb31, bb31, bb31, bb31) -> (bb31, bb31, bb31, bb31)
    %46 = arith.constant 3 : i32
    %47, %48, %49, %50 = func.call @ArcRf(%42, %43, %44, %45) : (bb31, bb31, bb31, bb31) -> (bb31, bb31, bb31, bb31)
    %51, %52, %53, %54 = func.call @SboxRf(%47, %48, %49, %50) : (bb31, bb31, bb31, bb31) -> (bb31, bb31, bb31, bb31)
    %55, %56, %57, %58 = func.call @MatMulRf(%51, %52, %53, %54) : (bb31, bb31, bb31, bb31) -> (bb31, bb31, bb31, bb31)
    func.return %55, %56, %57, %58 : bb31, bb31, bb31, bb31
  }
```
Inspecting the code, we see that the loop constants are present, but not used. 
Since we're all about performance, we should eliminate these. 

<mark>**Dead Code Elimination**.</mark> This is easy with the built in `dce(op)` pass imported `from xdsl.transforms.dead_code_elimination import dce`. 
Since this so-called Dead Code Elimination pass is applied on built-in types, they are automatically taken care of. 

Dead Code Elimination (DCE) is a commonly-used compiler optimization pass that is used to remove computations and branches that have no effect. 
As expected, the dead-end `ConstantOp` expressions are effectively taken care of by the `dce`. 

<mark>**Inlining.**</mark> Next-up, we can perform an Inlining pass. This replaces function calls with the function's body. 
By avoiding the call and return instructions, the program's performance is increased for small or frequently used functions. 
We wrote an `InlineFunctionCallsPattern(RewritePattern)` pass but there is likely a built-in way to do this with MLIR. 
We again don't expect to get much performance from inlining in our case, but it will allow us to illustrate more optimizations. 

Once we `print(module)` and inspect the code, or when we recall the lowered program from last week, we notice a gargantuan number of constants. 
Removing the duplicate constants should be handled by the dead code elimination, but likely due to our custom `mod_arith.constant` type, this does not happen. We might have to 'register' it somehow, or let it know that it behaves like a constant (or maybe something else even). 
We wrote a `RemoveDuplicateConstantsPattern(RewritePattern)` to replace a constant if its value is already previously declared in the program. 
This reduces the number of `ModConstantOp`s from 96 to just 11. 

The code of the functions that we inlined are still there in the module, we write another pass to remove these 'dead functions' but suspect there is a better way. 
Our `RemoveUnusedFunctionsPattern(RewritePattern)` was easy enough to write and removes function definitions that are never called, except for the `main` function, which is kept.

# 2. Other Generic Passes
Mid-end optimizations are a mature field, any many established compiler optimizations exist.
We list and describe some common mid-end optimizations that should amount to the 80% in the table below.

| Optimization                               | Purpose                                                                                       |
| ------------------------------------------ | --------------------------------------------------------------------------------------------- |
| **Loop Unrolling**                         | Expands loop bodies to reduce overhead and enable further optimizations                        |
| **Dead Code Elimination (DCE)**            | Removes computations and branches that have no effect                                          |
| **Inlining**                               | Replaces function calls with the body of the function (for small or frequently used functions) |
| **Common Subexpression Elimination (CSE)** | Avoids recomputing the same expression                                                         |
| **Strength Reduction**                     | Replaces expensive operations with cheaper ones (`x * 2 → x + x`)                              |
| **Constant Folding**                       | Evaluates constant expressions at compile time (`2 + 3 → 5`)                                   |
| **Constant Propagation**                   | Replaces variables with known constant values                                                  |
| **Copy Propagation**                       | Replaces copied variables with their source value (`x = y; z = x → z = y`)                     |
| **Loop Invariant Code Motion (LICM)**      | Moves calculations that don’t change inside loops outside of the loops                           |
| **Loop Fusion**                            | Combines adjacent loops over the same range to reduce overhead and improve cache locality      |
| **Induction Variable Simplification**      | Normalizes and simplifies loop induction variables                                               |
| **Branch Simplification**                  | Simplifies branches that always go one way (`if (true)` → just execute the body)                |
| **Loop Tiling (Blocking)**                 | Transforms loop access patterns to improve cache usage for large arrays                        |
| **Tail Call Elimination**                  | Converts certain recursive calls into loops                                                    |
| **Scalar Replacement of Aggregates**       | Breaks down structs/arrays into individual variables for better optimization                   |
| **Value Numbering (Local/Global)**         | Identifies equivalent expressions by assigning them unique identifiers                          |
| **Aggressive Dead Store Elimination**	     | Removes writes to memory that are never read later |

Not all of these generic passes are applicable in the elegantly simple Poseidon2 algorithm, and we leave looking into mid-end compiler passes that relate to the memory for another time. 
It is good to know the jargon and compiler lingo for when we watch one of those many presentations at one of the many (Euro)LLVM Developers's Meeting conferences. 


# 3. More Passes
To achieve performance, a compiler should consider the platform it is compiling to to some extent already in the mid-end. 
One optimization that illustrates a benefit here is a pass that looks for modular additions that follow modular multiplications. These can be fused together to form a Modular Multiply Accumulate operation. We added a `CollapseModArithMacPattern(RewritePattern)` to collapse `mod_arith.mul`s that are followed by `mod_arith.add`s into `ModMacOp` operations. 
The result is that two operations are combined into a single operation, reducing the code size and potentially the execution time of the program. Down the line, when lowered to `arith`, we might omit a modular reduction through some clever trick. 

Another more platform-aware optimization is a so-called Strength Reduction pass, which replaces expensive operations with cheaper ones. A multiplication can for instance be made cheaper using an addition or chain of additions when one operand is a small constant (`x * 2 -> x + x`). 
We added in a `StrengthReductionPattern(RewritePattern)` that performs strength reduction on multiplication operations by swapping them out with a `ModShlOp` shift operation or a combination of `ModShlOp` shifts and a `ModAddOp` addition. The patterns we cover are shown below. 
The expensive Montgomery multiplications and their associated conversion to the Montgomery domain and back are entirely avoided here, and we expect a significant reduction in `arith` operations as a result. 

```python3
 2·x → (x << 1)
 3·x → (x << 1) + x
 4·x → (x << 2)
 5·x → (x << 2) + x
 6·x → (x << 2) + (x << 1)
 7·x → (x << 3) - x
```
Our full compiler pass chain is shown below. 
```python3
# Mid-End Compiler Passes
def inline_passes(op: Operation):
    # Add rewrite patterns in this list
    rewrites = [
        UnrollForLoopPattern(),
        InlineFunctionCallsPattern(),
        RemoveDuplicateConstantsPattern(),
        RemoveUnusedFunctionsPattern(),
        CanonicalizationRewritePattern(), # A pass we haven't described yet
        CollapseModArithMacPattern(),
        StrengthReductionPattern(),
    ]

    PatternRewriteWalker(GreedyRewritePatternApplier(rewrites)).rewrite_module(
        op
    )
    # Apply Common Subexpression Eliminagtion & Dead Code Elimination
    cse(op)
    dce(op)
```

We can go even further here. When know the platform, its word sizes and which operations are cheap and which are expensive, we can start looking at known optimizations that relate to our datatypes and our algorithms. 
One such optimization is to lower `ModAddOp` operations not to full-blown modular operations, but to single `AddIOp` operations with a higher word size. The modular reduction can be applied at the very end, only when the higher word size would be overflown. A chain of modular additions and small modular shifts can be connected without 'interruption' of conditionals to keep the intermediate values canonical (i.e. within the range $0 \leq x < p$). 
This is a known optimization and is often referred to as **lazy modular arithmetic**. 

We could add lazy operations or other info to the dialect, but this does not align with MLIR's philosophy. The size of such a temporary accumulator is 'too much' of an implementation detail of the back-end, and not part of the mathematical model for modular arithmetic (integers mod p).
MLIR recommends to only add an attribute or type parameters if several different passes need the same information, when that's not the case, that kind of information is recommended to come out of an analysis that runs immediately before (or inside) the lowering pass rather than being stored on every value as an attribute.
A lowering should compute the width, and MLIR's dialect-conversion framework supports those patterns.
In other words, this is not an optimization pass but a lowering, and this particular one requires to look deep under the hood. We keep this pass for later. 


# 4. A/B Testing
We track the effect of our optimization passes by listing the frequency of each instruction in a bar chart before and after our optimization passes in the chart below. 
Doing this in a more fine-grained way, e.g. per optimization, is definitely more valuable, but that will be for another time. 
We upgraded our methodology from last week from manually to automatically counting the operations. 
Come to think of it, this might pass for an analysis pass if we set our bar for that definition low enough.

{{< figure src="/images/compiler_run_0_1.png" title="The Number of Operations after our First Lowering and after our First Optimizations" >}}

Some take-aways: 
- our optimizations bring us from 'that’s a lot of constants' to a significantly lower number since many constants were the same and are now only defined once;
- the number of multiplications is significantly reduced thanks to the strength reduction passes;
- the instructions needed to perform a modular reduction are reduced, again thanks to the strength reduction passes.

The other passes play a part as well, some to a larger extent, others to a more unnoticeable or lesser one. 
All in all, our optimization passes optimize, and not by a little. 
Morover, we are not out of optimizations and we expect even more reductions in instruction frequency after optimizing for lazy modular arithmetic. 

# Open Loops
Let's update and look at the list of open loops related to our Poseidon2 compiler. 

 - **Practical topics.**
 - [ ] Is there a better way to define matrices and vectors?
 - [ ] How can we visualize instruction dependencies in a graph?
 - [ ] <mark>**NEW!**</mark> How can we write analysis and optimization passes to enable Lazy Modular Arithmetic?
<br><br>
 - **Theory.**
 - [ ] What are traits in operations used for?
 - [ ] <mark>**NEW!**</mark> What are interfaces used for?
 - [ ] Does a type need an associated attribute like `IntegerType` has `IntegerAttr`?
 - [ ] How does xDSL/MLIR structure code? What can I learn from going through e.g. the `builtin` dialect?
 - [ ] What other ways are there to test operations and functions in a dialect? Is there a way that does not use an interpreter?

# Next-Up
We chalked up some wins this week. 
Our standard mid-end compiler optimization passes did their job and resulted in a ~73.6 % reduction in instruction count (or a ~3.8× shrink from 6739 to 1776 ops). 
Although there's likely a builtin way to do them, we vibe-coded and applied a number of optimization passes, and particularly, passes for unrolling loops, inlining function calls, removing duplicate constants, removing uncalled functions, and strength reduction. 
We actually added in an additional canonicalization pass already, which is not an optimization pass but does significantly help them. More on that next week. 

We got far, but we're not out of tricks just yet. We merely hit the limit of our understanding and we'll need another theory session next week to mitigate this. 
Hopefully with more illustrations and less text. See you there!

