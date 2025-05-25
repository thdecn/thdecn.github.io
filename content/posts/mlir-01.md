---
title: "Week 18"
date: 2025-05-04
description: "Specifying a Modular Arithmetic Type in MLIR."
keywords: ["mlir", "modular", "arithmetic", "tech", "blog"]
draft: false
tags: ["mlir"]
summary: Specifying a Modular Arithmetic Type in MLIR.
---

Following last week's blog post, we will look into more cleverly defining a type for performing our modular arithmetic. We specify a custom type where we can pass a value for the modulus rather than use type aliasing for each separate prime field. 
The custom type should bring some benefits; performing a modular addition on two operands should for instance fail when the two operands are from different prime fields. 
The end state is shown in the code below, which contrasts the builtin `IntegerType` with our custom `ModArithType`. 
```python
# Defining a Prime Field Type using Type Aliasing
bb31 = IntegerType(32)
m31 = IntegerType(32)
gl64 = IntegerType(64)

# Defining a Prime Field Type using an MLIR Type
bb31 = ModArithType([IntegerAttr(2**31-2**27+1, i32)])
m31 = ModArithType([IntegerAttr(2**31-1, i32)])
gl64 = ModArithType([IntegerAttr(2**64-2**32+1, i64)])
```
As in last week's writing, the sentences below are not necessarily correct yet. 
The primary goals is to consolidate my now third week of grappling with the compiler framework. 

# 1. Specifying a Modular Arithmetic Type.
In xDSL, we can create our type as follows. 
This code is adapted from the excellent [xDSL tutorial on defining dialects](https://xdsl.readthedocs.io/stable/marimo/defining_dialects/). 
```python
@irdl_attr_definition
class ModArithType(ParametrizedAttribute, TypeAttribute):
    name = "mod_arith.int"
    modulus: ParameterDef[IntegerAttr]
```
The `mod_arith.int` name is taken over from Google HEIR, and the modulus is defined as an integer through an `IntegerAttr`. 
Defining a type is now a lot more elegant (although just passing an `int` would be a lot more elegant still). 
```python
bb31 = ModArithType([IntegerAttr(2**31-2**27+1, i32)])
```
We can print the type using `print(bb31)` as well as some info on its properties using `print(bb31.modulus.value.data)` and `print(bb31.modulus.type)`. 
Our type prints to the following line. 
```mlir
!mod_arith.int<2013265921 : i32>
```

To tie an explicit value to the type, we need to define a constant operation. 
The constant operation turns a literal into an SSA value by defining a constant value via an attribute. Formal definitions of SSA values, types, and attributes are to come another time.
We provide the operation a helper constructor `from_int()` to pass an integer value and its modular type. I don't see yet how I can create a modular constant without somehow passing the `ModArithType`. 

```python
@irdl_op_definition
class ModConstantOp(IRDLOperation):
    name = "mod_arith.constant"

    value = attr_def(IntegerAttr)
    res = result_def(ModArithType)

    def __init__(self, value: IntegerAttr, value_type: ModArithType):
        super().__init__(result_types=[value_type], attributes={"value": value})

    @staticmethod
    def from_int(value: int, modulus: ModArithType) -> ConstantOp:
        return ModConstantOp(IntegerAttr(value, modulus.modulus.type), modulus)

    def get_type(self) -> ModArithType:
        # Constant cannot be unranked
        return cast(ModArithType, self.res.type)
```

We can define a constant with the `ModConstantOp` now.
```python
cst0 = ModConstantOp.from_int(123, bb31)
```
This gives us our MLIR operation, which we get using `print(cst0)`.
```mlir
%0 = "mod_arith.constant"() {value = 123 : i32} : 
	() -> !mod_arith.int<2013265921 : i32>
```
This looks different from the instantiation of a constant in HEIR, which looks like it does not require to pass the type. They use an `::mlir::TypedAttr` Typed Attribute instance for the value. I have no idea yet what this is though. Perhaps with time, our expressions will align. 


# 2. Adding Prime Field Elements.
We can adapt last week's `ModAddOp()` to work over our generic `mod_arith.int` type.
```python
@irdl_op_definition
class ModAddOp(IRDLOperation):
    # Modular addition operation
    name = "mod_arith.add"

    # No idea what traits are yet...
    traits = traits_def(Pure())

    # Define the operands and the result
    lhs = operand_def(ModArithType)
    rhs = operand_def(ModArithType)
    result = result_def(ModArithType)

    # The result type derives its type from the lhs operand
    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(operands=[lhs, rhs], result_types=[lhs.get_type()])
```
The result type derives its type from the `lhs` operand. 
There is no check on the types of the operands though, and nothing prevents us from generating nonsensical MLIR that adds elements from two different prime fields. 
```mlir
# Nonsensical MLIR that adds an element from the Baby Bear prime field 
# to an element from the Goldilocks prime field 
%0 = "mod_arith.add"(%1, %2) : (!mod_arith.int<2013265921 : i32>, 
	!mod_arith.int<18446744069414584321 : i64>) -> !mod_arith.int<2013265921 : i32>
``` 
We add a verify method to our `ModAddOp()` class to catch such behavior. 
```python
    def verify_(self) -> None:
        if not self.lhs.type == self.rhs.type:
            raise VerifyException(
                "Expected operand types to be equal: "
                f"{self.lhs.type}, {self.rhs.type}"
            )
```
For now, the `verify_()` is called manually as follows. Whether or not an automatic way exists to perform this check is left for another time. 
```python
add_bb31_gl64 = ModAddOp(bb31_cst, gl64_cst)
add_bb31_gl64.verify_()
```
Invoking the check leads to a clear and helpful error message.
```
raise VerifyException(
	xdsl.utils.exceptions.VerifyException: 
Expected operand types to be equal: 
!mod_arith.int<2013265921 : i32>, !mod_arith.int<18446744069414584321 : i64>
```

# 3. Adding More Operations.
Aside from a type and a modular addition operation, HEIR's `mod_arith` dialect defines several more operations.
 - The `mod_arith.constant`, `mod_arith.add`, `mod_arith.sub`, `mod_arith.mul`, and `mod_arith.mac` provide a way to create a constant as well as to perform commonly used arithmetic operations;
 - the `mod_arith.barrett_reduce`, `mod_arith.reduce`, and `mod_arith.subifge` help in keeping the value of the type canonical, i.e. in the right range;
 - the `mod_arith.encapsulate`, `mod_arith.extract`, and `mod_arith.mod_switch` help convert from one type to another, whether from and to an integer in the case of the first two, or to another prime field in the third one. 

We could go for a one-to-one mapping with HEIR by implementing all their operations, but for now, we won't implement the reduction operations and assume we are always working with canonical values. 
Furthermore, we foresee adding `mod_arith.inverse` and `mod_arith.pow` operations or a way to enable lazy modular reductions (which might need a different dialect?). 
We will therefore deviate from HEIR sooner rather than later and won't be too pedantic here as a result. 
 
Once the remaining operations are implemented, we have a very expressive language for modular arithmetic!
 
```mlir
builtin.module {
  func.func @main(%0 : i32, %1 : !mod_arith.int<2013265921 : i32>) -> i32 {
    %2 = "mod_arith.encapsulate"(%0) : (i32) -> !mod_arith.int<2013265921 : i32>
    %3 = "mod_arith.constant"() {value = 123 : i32} : () -> !mod_arith.int<2013265921 : i32>
    %4 = "mod_arith.add"(%1, %2) : (!mod_arith.int<2013265921 : i32>, !mod_arith.int<2013265921 : i32>) -> !mod_arith.int<2013265921 : i32>
    %5 = "mod_arith.mul"(%3, %4) : (!mod_arith.int<2013265921 : i32>, !mod_arith.int<2013265921 : i32>) -> !mod_arith.int<2013265921 : i32>
    %6 = "mod_arith.sub"(%3, %4) : (!mod_arith.int<2013265921 : i32>, !mod_arith.int<2013265921 : i32>) -> !mod_arith.int<2013265921 : i32>
    %7 = "mod_arith.constant"() {value = 0 : i32} : () -> !mod_arith.int<2013265921 : i32>
    %8 = "mod_arith.mac"(%5, %6, %7) : (!mod_arith.int<2013265921 : i32>, !mod_arith.int<2013265921 : i32>, !mod_arith.int<2013265921 : i32>) -> !mod_arith.int<2013265921 : i32>
    %9 = "mod_arith.extract"(%8) : (!mod_arith.int<2013265921 : i32>) -> i32
    func.return %9 : i32
  }
}
```

# Open Loops
The more light we shine, the more shadows we uncover, and we're left with a list of open questions (some rephrased from last week and some new ones).

 - [X] [How do we run our modules?]({{< relref "mlir-02.md" >}})
 - [X] [What are the formal definitions and uses of Modules, Blocks, Regions, SSA Values, Operations, Attributes, Types, and Properties? What other structures are there?]({{< relref "mlir-03.md" >}})
 - [X] [How do we lower one module to another?]({{< relref "mlir-04.md" >}})
 - [ ] <mark>**NEW!**</mark> How do we set default values or default types? 
 - [ ] <mark>**NEW!**</mark> Our `mod_arith.int` type is a bit verbose, can we print a string rather than a value to clarify the prime? 
 - [ ] <mark>**NEW!**</mark> How do we constraint the signedness and canonicalization of the inputs?
 - [ ] <mark>**NEW!**</mark> Does a type need an associated attribute like `IntegerType` has `IntegerAttr`?
 - [ ] <mark>**NEW!**</mark> How does xDSL/MLIR structure code? What can I learn from going through e.g. the `builtin` dialect?


# Next-Up
Our codebase is expanding, and it's time to test some--or ideally all--of our code. 
We will need to run our modules and investigate their correct input/output behavior. 
That's the topic we'll be looking into next week. 




