---
title: "Expressing a Poseidon Hash in MLIR"
date: 2025-05-31
description: "Expressing a Poseidon Hash in MLIR."
math: true
keywords: ["mlir", "modular", "arithmetic", "poseidon", "hashing", "zk", "tech", "blog"]
draft: false
tags: ["mlir", "zk"]
summary: Week 22, Leuven --- Messing around with a toy application.
---

Over the past weeks, we got the basics on MLIR down through the more forgiving xDSL Python library. 
We got through a compiler loop, and more specifically, a lowering of our custom `mod_arith` dialect for modular arithmetic to the MLIR-native `arith` dialect, and tested the correctness of the results with an interpreter. 

It's time to get overconfident and look at a real-world example to implement.
Hopefully, by amping up our example's complexity, we break a thing or two and learn more about MLIR by fixing and expanding our dialect and compiler. 

Aside from lowering, compilers define analysis passes and optimization passes. 
In the coming weeks, over the next set of blog posts, we'll look deeper into implementing those passes. 
As a substantial example to analyze and optimize, we choose the Poseidon2 hash function that is commonly used in Proof Systems. 
As it is often a performance bottleneck, and as its computation requires plenty of modular arithmetic operations, it is the ideal candidate for our optimization purposes over our `mod_arith` dialect. <!--That and some other reasons.-->

# 1. The Poseidon2 Hash Function
The Poseidon2 is the successor of the Poseidon hash function, and has been adopted by the most commonly used Proof Systems (including Polygon's Plonky3, Succinct's SP1, and Risc Zero's risc0 zkVM). 
Poseidon2 follows the classic Substitution-Permutation Network (SPN) design idea. 
It distinguished between applying full and partial layers to `t` state elements, where the partial layers are computationally lighter than the full layers. 
Each layer consists of three round functions: an Add Round Constant (ARC) function, a Sub Words function, and Mix function. 
The ARC layer adds a random constant to all state element. 
The Sub Words layer applies a power map S-box $x \mapsto x^{d}$ to either all state elements in the full rounds, or to a single state element in the partial ones. The remaining `(t-1)` state elements are left unaltered in the partial layer. 
The Mix layer multiplies the state elements with a `t x t`-matrix defined in the [specification](https://eprint.iacr.org/2023/323.pdf). 
When the state size is $t=8$ or $t=12$, the external round matrix $M_{\varepsilon}$ is set as follows. 

$$
M_{\varepsilon} = \text{circ}(2 \cdot M_4, M_4, \ldots , M_4) \in \mathbb{F}_p^{t \times t} \\; 
\text{with} \\; 
M_4 = 
\begin{bmatrix}
 5 & 7 & 1 & 3 \\\\
 4 & 6 & 1 & 1 \\\\
 1 & 3 & 5 & 7 \\\\
 1 & 1 & 4 & 6
\end{bmatrix}
$$

For our purposes, we will implement a toy Poseidon2, and only implement $R_f=4$ full rounds, over $t=4$ state elements. We choose 4 random ARC layer constants that stay the same for each round, a power map of degree $d=3$ for the S-box layer, and the $M_4$ matrix for the Mix layer. 

{{< figure src="/images/Poseidon2_Rf.png" title="A Full Round of a State-Reduced Poseidon2 Hash Function" >}}

Poseidon2 is a so-called Arithmetization-Oriented hash functions, and many other ones exist; e.g. Rescue-Prime, Tip5, and Monolith to name a few.

# 2. The MLIR Toy Poseidon2 Module
Implementing the module is not too daunting thanks to our custom dialect (and thanks to Poseidon's elegantly regular structure).
For our own comfort, and for making the following lines of code more readable, we will make it even easier by starting out with type alias our Baby Bear prime. 
```python3
bb31 = ModArithType([IntegerAttr(2**31-2**27+1, i32)])
```
Printing this type will give the verbose `!mod_arith.int<2013265921 : i32>`, which we can reduce to a much nicer `bb31` by adapting the `xdsl/xdsl/printer.py` file as follows. (This file might not be the place to put this code, and I expect it should be in the dialect file, but it works for now, and solving this can wait until I try to upstream my code.
```python3
    def print_attribute(self, attribute: Attribute) -> None:
    	# ...
        if attribute.__class__.__name__ == "ModArithType":
            if hasattr(attribute, 'modulus') and 
                attribute.modulus.value.data == 2**31-2**27+1:
                self.print_string("bb31")
            else:
                self.print_string(
                    "!mod_arith.int<" + f"{attribute.modulus.value.data}" 
                    + " : " + f"{attribute.modulus.type}" + ">")
            return
        # ...
```
With this quick plumbing (which was on our open loops list), we continue by defining our module and functions using the `Builder`. We use a Python For Loop here to loop over the state vector and insert the right modular operations. We can alternatively use the For Loop offered by the `scf` dialect. We will mix both in our module, and comment on the differences down the line. 
For the matrix in the MatMul layer, we store all values as constants. Here too, there is likely a better way to do this. 
```python3
# Create a module, and create a builder entry point
#  at the beginning of its only block
module = ModuleOp([])
builder = Builder(InsertPoint.at_end(module.body.block))

# Define the Full Round ARC Layer Function "ArcRf"
fullRoundARC = FuncOp("ArcRf", 
  ([bb31, bb31, bb31, bb31], [bb31, bb31, bb31, bb31]))
builder.insert(fullRoundARC)
builder.insertion_point = InsertPoint.at_end(fullRoundARC.body.block)
ArcCsts = [ModConstantOp.from_int(259, bb31), ModConstantOp.from_int(258, bb31),
		   ModConstantOp.from_int(257, bb31), ModConstantOp.from_int(256, bb31)]
stateOut = []
for i in range(4):
    builder.insert(ArcCsts[i])
    # Add the round constant
    stateOuti = ModAddOp(fullRoundARC.args[i], ArcCsts[i])
    builder.insert(stateOuti)
    stateOut.append(stateOuti)
builder.insert(ReturnOp(stateOut[0], stateOut[1], stateOut[2], stateOut[3]))
builder = Builder(InsertPoint.at_end(module.body.block))

# Define the Full Round Sbox Layer Function "SboxRf"
fullRoundSbox = FuncOp("SboxRf", 
  ([bb31, bb31, bb31, bb31], [bb31, bb31, bb31, bb31]))
builder.insert(fullRoundSbox)
builder.insertion_point = InsertPoint.at_end(fullRoundSbox.body.block)
stateOut = []
for i in range(4):
    # Compute x**2
    sqStatei = ModMulOp(fullRoundSbox.args[i], fullRoundSbox.args[i])
    builder.insert(sqStatei)
    # Compute x**3
    cuStatei = ModMulOp(fullRoundSbox.args[i], sqStatei)
    builder.insert(cuStatei)
    stateOut.append(cuStatei)
builder.insert(ReturnOp(stateOut[0], stateOut[1], stateOut[2], stateOut[3]))
builder = Builder(InsertPoint.at_end(module.body.block))

# Define the Full Round Mix MatMul Function "MatMulRf"
fullRoundMatMul = FuncOp("MatMulRf", 
  ([bb31, bb31, bb31, bb31], [bb31, bb31, bb31, bb31]))
builder.insert(fullRoundMatMul)
builder.insertion_point = InsertPoint.at_end(fullRoundMatMul.body.block)
# Define the Matrix
M4 = [
        [ModConstantOp.from_int(5, bb31), ModConstantOp.from_int(7, bb31),
         ModConstantOp.from_int(1, bb31), ModConstantOp.from_int(3, bb31)],
        [ModConstantOp.from_int(4, bb31), ModConstantOp.from_int(6, bb31),
         ModConstantOp.from_int(1, bb31), ModConstantOp.from_int(1, bb31)],
        [ModConstantOp.from_int(1, bb31), ModConstantOp.from_int(3, bb31),
         ModConstantOp.from_int(5, bb31), ModConstantOp.from_int(7, bb31)],
        [ModConstantOp.from_int(1, bb31), ModConstantOp.from_int(1, bb31),
         ModConstantOp.from_int(4, bb31), ModConstantOp.from_int(6, bb31)]
    ]
# Store the matrix as constants
for i in range(4):
    for j in range(4):
        builder.insert(M4[i][j])
# Initialize the output state vector
stateOut = [ModConstantOp.from_int(0, bb31), ModConstantOp.from_int(0, bb31),
            ModConstantOp.from_int(0, bb31), ModConstantOp.from_int(0, bb31)]
for i in range(4):
    builder.insert(stateOut[i])
# Compute the output
for i in range(4):
    for j in range(4):
        mulAB = ModMulOp(fullRoundMatMul.args[j], M4[i][j])
        builder.insert(mulAB)
        stateOut[i] = ModAddOp(stateOut[i], mulAB)
        builder.insert(stateOut[i])
builder.insert(ReturnOp(stateOut[0], stateOut[1], stateOut[2], stateOut[3]))
builder = Builder(InsertPoint.at_end(module.body.block))
```
That's a lot of code to put in a snippet, and there's even more to come. 
The full round calls each of these functions in a For Loop.
We use the For Loop `scf.for` from the `scf` dialect this time, as well as the function calls `func.calls` from the `func` dialect. 
```python3
# Create the Main Full Round function
fullRoundMain = FuncOp("main", ([bb31, bb31, bb31, bb31], [bb31, bb31, bb31, bb31]))
builder.insert(fullRoundMain)
builder.insertion_point = InsertPoint.at_end(fullRoundMain.body.block)
## Create a For Loop
lowerBound = ConstantOp(IntegerAttr(0, i32))
upperBound = ConstantOp(IntegerAttr(4, i32))
step = ConstantOp(IntegerAttr(1, i32))
builder.insert(lowerBound)
builder.insert(upperBound)
builder.insert(step)
# Define the initial values of the iteration arguments
# Create a new Block and Builder to populate the Block
# The i32 type holds the iterator index, 
#  the 4 bb31 types hold the iteration arguments
loopBody = Block(ops=[], arg_types=[i32, bb31, bb31, bb31, bb31])
loopBuilder = Builder(InsertPoint.at_end(loopBody))
iterArgs = [fullRoundMain.args[0], fullRoundMain.args[1], 
            fullRoundMain.args[2], fullRoundMain.args[3]]
funcCallArc = loopBuilder.insert(CallOp(
		"ArcRf", 
		[loopBody.args[1], loopBody.args[2], 
		 loopBody.args[3], loopBody.args[4]], 				
		[bb31, bb31, bb31, bb31]))
funcCallSbox = loopBuilder.insert(CallOp(
		"SboxRf", 
		[funcCallArc.results[0], funcCallArc.results[1], 
		 funcCallArc.results[2], funcCallArc.results[3]], 
		[bb31, bb31, bb31, bb31]))
funcCallMatMul = loopBuilder.insert(CallOp(
		"MatMulRf", 
		[funcCallSbox.results[0], funcCallSbox.results[1], 
		 funcCallSbox.results[2], funcCallSbox.results[3]], 
		[bb31, bb31, bb31, bb31]))
# End the For Loop with a YieldOp
loopBuilder.insert(YieldOp(funcCallMatMul.results[0], funcCallMatMul.results[1], 
						   funcCallMatMul.results[2], funcCallMatMul.results[3]))
forLoop = ForOp(lowerBound, upperBound, step, iterArgs, loopBody)
builder.insert(forLoop)
builder.insert(ReturnOp(forLoop.results[0], forLoop.results[1], 
						forLoop.results[2], forLoop.results[3]))
```

The two---for us---new MLIR mechanisms here are a For Loop and a function call operation. 
The For Loop is somewhat tricky and requires us to define a new Block, with argument types that correspond to the For Loop's iteration index (think `i`) and to the the variables that get updated and carried over to the next loop iteration. 
In our case, we have one iteration index (`i32`) and four state elements of type `bb31` to keep track of. 
After assigning the initial values to the iteration arguments `iterArgs` and populating our loop body, we conclude the loop body with a `YieldOp()` to return the results of our loop. 
These correspond to the iteration arguments and define the input values to the next loop iteration. 

The loop body itself is populated with the `CallOp()` operations that define Poseidon2's full round, but state-reduced, functionality. 


We'll `print(module)` to see the result once. Strap yourselves in! This isn't the prettiest...
```mlir
builtin.module {
  func.func @ArcRf(%0 : bb31, %1 : bb31, %2 : bb31, %3 : bb31) 
  					-> (bb31, bb31, bb31, bb31) {
    %4 = "mod_arith.constant"() {value = 259 : i32} : () -> bb31
    %5 = "mod_arith.add"(%0, %4) : (bb31, bb31) -> bb31
    %6 = "mod_arith.constant"() {value = 258 : i32} : () -> bb31
    %7 = "mod_arith.add"(%1, %6) : (bb31, bb31) -> bb31
    %8 = "mod_arith.constant"() {value = 257 : i32} : () -> bb31
    %9 = "mod_arith.add"(%2, %8) : (bb31, bb31) -> bb31
    %10 = "mod_arith.constant"() {value = 256 : i32} : () -> bb31
    %11 = "mod_arith.add"(%3, %10) : (bb31, bb31) -> bb31
    func.return %5, %7, %9, %11 : bb31, bb31, bb31, bb31
  }
  func.func @SboxRf(%0 : bb31, %1 : bb31, %2 : bb31, %3 : bb31) 
  					-> (bb31, bb31, bb31, bb31) {
    %4 = "mod_arith.mul"(%0, %0) : (bb31, bb31) -> bb31
    %5 = "mod_arith.mul"(%0, %4) : (bb31, bb31) -> bb31
    %6 = "mod_arith.mul"(%1, %1) : (bb31, bb31) -> bb31
    %7 = "mod_arith.mul"(%1, %6) : (bb31, bb31) -> bb31
    %8 = "mod_arith.mul"(%2, %2) : (bb31, bb31) -> bb31
    %9 = "mod_arith.mul"(%2, %8) : (bb31, bb31) -> bb31
    %10 = "mod_arith.mul"(%3, %3) : (bb31, bb31) -> bb31
    %11 = "mod_arith.mul"(%3, %10) : (bb31, bb31) -> bb31
    func.return %5, %7, %9, %11 : bb31, bb31, bb31, bb31
  }
  func.func @MatMulRf(%0 : bb31, %1 : bb31, %2 : bb31, %3 : bb31) 
  			-> (bb31, bb31, bb31, bb31) {
    %4 = "mod_arith.constant"() {value = 5 : i32} : () -> bb31
    %5 = "mod_arith.constant"() {value = 7 : i32} : () -> bb31
    %6 = "mod_arith.constant"() {value = 1 : i32} : () -> bb31
    %7 = "mod_arith.constant"() {value = 3 : i32} : () -> bb31
    // ...
    %15 = "mod_arith.constant"() {value = 7 : i32} : () -> bb31
    %16 = "mod_arith.constant"() {value = 1 : i32} : () -> bb31
    %17 = "mod_arith.constant"() {value = 1 : i32} : () -> bb31
    %18 = "mod_arith.constant"() {value = 4 : i32} : () -> bb31
    %19 = "mod_arith.constant"() {value = 6 : i32} : () -> bb31
    %20 = "mod_arith.constant"() {value = 0 : i32} : () -> bb31
    %21 = "mod_arith.constant"() {value = 0 : i32} : () -> bb31
    %22 = "mod_arith.constant"() {value = 0 : i32} : () -> bb31
    %23 = "mod_arith.constant"() {value = 0 : i32} : () -> bb31
    %24 = "mod_arith.mul"(%0, %4) : (bb31, bb31) -> bb31
    %25 = "mod_arith.add"(%20, %24) : (bb31, bb31) -> bb31
    %26 = "mod_arith.mul"(%1, %5) : (bb31, bb31) -> bb31
    %27 = "mod_arith.add"(%25, %26) : (bb31, bb31) -> bb31
    // ...
    %52 = "mod_arith.mul"(%2, %18) : (bb31, bb31) -> bb31
    %53 = "mod_arith.add"(%51, %52) : (bb31, bb31) -> bb31
    %54 = "mod_arith.mul"(%3, %19) : (bb31, bb31) -> bb31
    %55 = "mod_arith.add"(%53, %54) : (bb31, bb31) -> bb31
    func.return %31, %39, %47, %55 : bb31, bb31, bb31, bb31
  }
  func.func @main(%0 : bb31, %1 : bb31, %2 : bb31, %3 : bb31) 
  					-> (bb31, bb31, bb31, bb31) {
    %4 = arith.constant 0 : i32
    %5 = arith.constant 4 : i32
    %6 = arith.constant 1 : i32
    %7, %8, %9, %10 = scf.for %11 = %4 to %5 step %6 
    					iter_args(%12 = %0, %13 = %1, %14 = %2, %15 = %3) 
    					-> (bb31, bb31, bb31, bb31)  : i32 {
      %16, %17, %18, %19 = func.call @ArcRf(%12, %13, %14, %15) : 
      						(bb31, bb31, bb31, bb31) -> (bb31, bb31, bb31, bb31)
      %20, %21, %22, %23 = func.call @SboxRf(%16, %17, %18, %19) : 
      						(bb31, bb31, bb31, bb31) -> (bb31, bb31, bb31, bb31)
      %24, %25, %26, %27 = func.call @MatMulRf(%20, %21, %22, %23) : 
      						(bb31, bb31, bb31, bb31) -> (bb31, bb31, bb31, bb31)
      scf.yield %24, %25, %26, %27 : bb31, bb31, bb31, bb31
    }
    func.return %7, %8, %9, %10 : bb31, bb31, bb31, bb31
  }
}
```
While this is very verbose and while there are ways to reduce the number of lines by for instance using more For Loops, our pattern-recognizing brains do get to work when inspecting this big chungus. 
We will come back to some of these patterns in the coming weeks. 

# 3. Compiling the ModMulOp to the Arith Dialect
To lower our toy Poseidon2 model from `mod_arith` to `arith` we need to define the lowering of the `ModMulOp` operation. 
We use the [Montgomery modular multiplication](https://en.wikipedia.org/wiki/Montgomery_modular_multiplication). While I'll omit the deep dive on the 'how', the 'what does it look like in MLIR' is shown below. 
 
```mlir
func.func @MontgomeryMult(%0 : i32, %1 : i32) -> i32 {
  %2 = arith.extui %0 : i32 to i64
  %3 = arith.extui %1 : i32 to i64
  %4 = arith.muli %2, %3 : i64
  %5 = arith.constant 2013265919 : i64
  %6 = arith.constant 4294967295 : i64
  %7 = arith.muli %4, %5 : i64
  %8 = arith.andi %7, %6 : i64
  %9 = arith.constant 2013265921 : i64
  %10 = arith.constant 32 : i64
  %11 = arith.muli %8, %9 : i64
  %12 = arith.addi %4, %11 : i64
  %13 = arith.shrui %12, %10 : i64
  %14 = arith.cmpi uge, %13, %9 : i64
  %15 = arith.subi %13, %9 : i64
  %16 = arith.select %14, %15, %13 : i64
  %17 = arith.trunci %16 : i64 to i32
  func.return %17 : i32
}
func.func @ModMult(%0 : i32, %1 : i32) -> i32 {
  %2 = arith.constant 1172168163 : i32
  %3 = func.call @MontgomeryMult(%0, %2) : (i32, i32) -> i32
  %4 = func.call @MontgomeryMult(%1, %2) : (i32, i32) -> i32
  %5 = func.call @MontgomeryMult(%3, %4) : (i32, i32) -> i32
  %6 = arith.constant 1 : i32
  %7 = func.call @MontgomeryMult(%5, %6) : (i32, i32) -> i32
  func.return %7 : i32
}
```
There are quite some new operations we used here. Aside from the straightforward `ConstantOp()`, `AddiOp()`, `SubiOp()`, `MuliOp()`, `CmpiOp()`, and `SelectOp()` from the `arith` dialect, and the `FuncOp()`, `ReturnOp()`, and `CallOp()` from the `func` dialect, we have use the following ops in our lowering. 
A brief explanation is given here, and the full details can be found in the [dialect's doc](https://mlir.llvm.org/docs/Dialects/ArithOps/). 
- `ExtUIOp()`. Zero-extends an integer with a lower bit-width to an integer with a higher bit-width by inserting zero bits in the MSBs.
- `AndIOp()`. Performs a bitwise AND operation between two integers of the same type. 
- `ShRUIOp()`. Right-shifts an unsigned integer with the value of another unsigned integer.
- `TruncIOp().` Trucates an integer from a wider type to a type with less bits.

# 4. What Makes a Good Compiler?
The goal of a compiler, is to take code with a set of human-friendly characteristics (e.g. easy to read, easy to write, easy to maintain) and to translate or transform it into some code that has the same functionality but a different set of machine-friendly characteristics (e.g. fits in the machine's RAM, optimally uses the machine's instructions, does not consume excessive power when running the code). 

We'll consider the human-friendly characteristics to be the domain of programming language designers and we'll leave it with that to the professionals. 
A good compiler can then be evaluated based on the machine code or lowest operations it compiles to. 
Depending on the target platform, we can evaluate the code across several dimensions.
- **Program Size.** How large is the size of the program in G/M/K/Bytes? This is important when compiling to IoT processors with small memory footprints, as well as to keep instructions as much as possible in a cache (close to the compute).
- **Program Performance.** Performance can be measured in 'useful instructions per clock cycle'. The smaller the number of instructions, and the smaller the number of pipeline stalls between them, or more completely, the smaller the number of clock cycles required, the faster the program will execute. 

Other metrics we will not discuss here are the program's power consumption, its code portability and target coverage, its compile time, its security, or the memory footprint of the runtime. Many more exist.

In our case, the lowest-level operations are currently operations from the `arith` dialect. 
There are currently two metrics we can straightforwardly count: the number of instructions and their type (some instructions might be more expensive and take longer to execute than others). We can go in more depth and assign weights to each instruction type, as well as track dependencies between consecutive instructions later on, after we acquire the skills to do so. 

For now, we just list them in a bar chart with the frequency of each instruction after our first lowering pass. We omit the function calls here and multiply the ops inside the For Loops by the times they are executed. This is currently done manually, exo-MLIR, and we'll look into automating this as part of an analysis pass soon enough.
First take-away: **that's a lot of constants**. 

{{< figure src="/images/compiler_run_0.png" title="The Number of Operations After our First Lowering" >}}


# Open Loops
Rather than re-listing all open loops we collected over our introductory sessions, I'll only explicitly list the items I expect we'll need to optimize our Poseidon2 compiler. 
To further compress the list, we'll keep chipping away at the category of 'Small Changes' without listing them every time. 

 - **Practical topics.**
 - [ ] How do we estimate the performance of a module?
 - [ ] <mark>**NEW!**</mark> Is there a better way to define matrices and vectors?
 - [ ] <mark>**NEW!**</mark> How can we visualize instruction dependencies in a graph?
<br><br>
 - **Theory.**
 - [ ] What are traits in operations used for?
 - [ ] What other ways are there to test operations and functions in a dialect? Is there a way that does not use an interpreter? 
 - [ ] What is the elementary subset of compiler terminology to be productive? What is e.g. Dead Code Elimination (DCE), Common Subexpression Elimination (CSE), and Loop-Invariant Code Motion (LICME). What is a Definition-Use (DU) Chain? What are PHI nodes, what are alternatives to SSA representation? 

# Next-Up
We had some lolz this week, lots of lines of code. 
We wrote a full round of a reduced-state Poseidon2 hash function in MLIR, compiled it down to operations from the `arith` dialect, and learned `scf` For Loops and `func` Function Calls along the way (and definitely underused them). 

Counting the ops, we tally a full round of the reduced-state Poseidon2 hash to require thousands of `arith` operations. 
This is perfect; that means we have an excuse to study some compiler analysis & optimization passes!

