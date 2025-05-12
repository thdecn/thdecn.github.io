---
title: "Week 19"
date: 2025-05-05
description: "Running MLIR Code."
keywords: ["mlir", "modular", "arithmetic", "tech", "blog"]
draft: false
tags: ["mlir"]
summary: Running MLIR Code.
---

In our first blog, we intuited that writing modular arithmetic in MLIR's `arith` dialect would have some benefits. 
We used xDSL to write a modular addition operation for the Baby Bear prime field using basic arithmetic and control flow, which gave us some readable MLIR code. 
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
Now it turns out that there was something true about that intuition; the `arith` dialect can be **interpreted** in xDSL. 
This means we can provide inputs to our code snippet, observe the outputs, and debug until we get the modular addition right. 
In what follows, we will fire up this interpreter (and likely do some debugging as well). 


<!-- a first PR? -->


The common warning applies here as well; the sentences below are not necessarily correct, and the interpreter might not be the 'right' way to run and check the code either. 
The primary goals is to consolidate what worked for me in my fourth week of grappling with the compiler framework. 


# 1. Creating Test Vectors
We'll start off by generating test vectors for the modular addition. 
We can of course just use `(a + b) % mod`, with the warning that the result might not be Euclidian, i.e. not in the canonical range. 
We'll instead wrap it in a function, making it easier to reuse as a template for more complicated functions. 
```python
def mod_add_check(a: int, b: int, mod: int) -> int:
    """
    Python implementation of mod_add
    """
    result = (a + b) % mod
    return result
```

# 2. Creating a Test Module
We use a builder to create the `mod_add` function that we'll be testing.
This time, we use an `ImplicitBuilder`, a different approach taken over from some `xdsl/tests/interpreters/test_*.py` tests. 
It's taken over, so in other words, I have no idea yet how this works or why one would use this approach over another.
It looks for instance that a Region is created, a concept we haven't defined yet with clarity. 
Adapting it to our case however does lead to the expected MLIR code, aside from a small change; the modulus is now taken in as a third input. 
```python
@ModuleOp
@Builder.implicit_region
def mod_add_op():
    # Create a function
    func = FuncOp("mod_add", ((i32, i32, i32), (i32,)))
    with ImplicitBuilder(func.body) as b:
        # Create the possible output values of the mod operation
        res_no_overflow = arith.AddiOp(b[0], b[1])
        res_overflow = arith.SubiOp(res_no_overflow, b[2])
        # Create the selector to choose the right output value 
        is_lt = arith.CmpiOp(res_no_overflow, b[2], "ult")
        res_select = arith.SelectOp(is_lt, res_no_overflow, res_overflow)
        # Create a return statement
        ReturnOp(res_select)
```
```mlir
func.func @mod_add(%0 : i32, %1 : i32, %2 : i32) -> i32 {
  %3 = arith.addi %0, %1 : i32
  %4 = arith.subi %3, %2 : i32
  %5 = arith.cmpi ult, %3, %2 : i32
  %6 = arith.select %5, %3, %4 : i32
  func.return %6 : i32
}
```
# 3. Create an Interpreter
Next, we create an `Interpreter`. This is again taken over from a preexisting test that uses the interpreter. It is somewhat more clear what is going on here though. 
We first verify that the module we pass is (syntactically) ok. 
We then create the interpreter for the `mod_add` operation and register the dialects used by the module (`func` and `arith`). 
Calling the interpreter with our inputs then gives back our result.
The `func_name` should match the name of the test function we defined in the previous section. 
```python
def modadd_interp(
		module_op: ModuleOp, func_name: str, 
		a: int, b: int, mod: int) -> int:
    module_op.verify()
    interpreter = Interpreter(module_op)
    interpreter.register_implementations(FuncFunctions())
    interpreter.register_implementations(ArithFunctions())
    (result,) = interpreter.call_op(func_name, (a, b, mod))
    return result
```

# 4. Creating the Full Test Function
We can stitch everything together into the code below to test multiple random test vectors. 
```python
def test_mod_add():
    mod = 2**31 - 2**27 + 1    
    for i in range(10):
        a = random.randint(0, 2**32 - 1) % mod
        b = random.randint(0, 2**32 - 1) % mod
        res = mod_add_check(a, b, mod)
        received_res = modadd_interp(mod_add_op, "mod_add", a, b, mod)
        assert res == received_res
```
Due to the size of our input space, it would take quite a bit of time to test all possible input combinations exhaustively. Using random test vectors helps us maximize our luck surface area for catching bugs. 
From personal extensive experience with catching bugs in verilog implementations of modular arithmetic circuits, not testing a significant number of inputs does make me feel somewhat squeamish. 
Fortunately, there is a way we can still perform an exhaustive sanity check on our code; we just have to choose a smaller modulus where an exhaustive test does work. 
To scale the test down correctly, we have two changes to make:
1. Choose a smaller modulus; we scale down from `bb31` to `m7`, the Mersenne prime that satisfies `m3 = 2**3-1 = 7`. 
2. Choose a smaller integer element to represent our operands and (intermediate) results; we reduce the size of our integers from `i32` to `i4`. 

The second change is to check that any unexpected behavior from overflows translates as well. 
If not, we'll have to change some of our code by debugging. 
The test passes after putting in our changes, increasing our confidence that the modular addition operation we've written using the `arith` dialect is not faulty. 


# Open Loops
With every answer we find, more questions emerge, and we're adding some more open questions to our list. 

 - [ ] What are the formal definitions and uses of Modules, Blocks, Regions, SSA Values, Operations, Attributes, Types, and Properties? What other structures are there? 
 - [ ] How do we lower one module to another? 
 - [ ] How do we set default values or default types? 
 - [ ] Our `mod_arith.int` type is a bit verbose, can we print a string rather than a value to clarify the prime? 
 - [ ] How do we constraint the signedness and canonicalization of the inputs?
 - [ ] Does a type need an associated attribute like `IntegerType` has `IntegerAttr`?
 - [ ] How does xDSL/MLIR structure code? What can I learn from going through e.g. the `builtin` dialect?
 - [ ] <mark>**NEW!**</mark> What other ways are there to test operations and functions in a dialect? Is there a way that does not use an interpreter? 
 - [ ] <mark>**NEW!**</mark> What are the different ways to create MLIR code? How does the `Builder` compare to the `ImplicitBuilder`? 
 - [ ] <mark>**NEW!**</mark> How do we catch overflows in the `arith` dialect?


# Next-Up
Now that we can convince ourselves that what we've written in the `arith` dialect matches the expected behavior of our `mod_arith` dialect, 
it's time to investigate how we can convert---or lower--- `mod_arith` to `arith`. 
Before we look into that essential compiler topic, we'll have to take a deep dive into some MLIR theory, and in particular, the IR structure. That's the topic for next week. 



