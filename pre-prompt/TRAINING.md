# Prompt
Your an expert prompt writer, to write a detailed and crips prompt to be fed as input to a prompt generator.
Ultimate goal of the generated prompt to make a python script to produce training and testing data with sufficient noise.

## Entities 

### Actors
15 well known / common male names.
15 well known / common female names.
Equally distributed from hinud, muslim and christain faith.

Actors are school children / teacher and friends.

### Relationship
1:1 
Actor 2 is [friend, teacher, student, senior, junior] of the Actor 1

### Verbs / Actions 

#### Addition
get, add, receive
### subtraction
give, pay, gift

## Assets and units 
Assets are expressed in same unit, [$, Rupee (Add the symbol), marbles, candies, books, pencils]
treat .*[UNIT].* 100 same as 100 .*[UNIT].* (I am using regular expression convention)
example $10 === 10 $ == 10$ === $ 10
apply this rule all assets lited above.
### Number of operands and target operation
1 oparand - always add
prompt: Mahesh have 3 cars.
operation: {
    kind: add
    operands:[3,]
}

2 operands - add / subtract based on rule given above
prompt:
John had 45$ and he got $10 from Mahesh
operation: {
    kind: add
    operands:[45,10]
}

John had 45$ and he gave $10 from Mahesh
operation: {
    kind: subtract
    operands:[45,10]
}

3 - always insert add: [get, add, receive] and result is sum of 
3 operands - sum on rule given above

There are 10 champa, 6 rose, 6 jasmine, 10 marigold flowers
operation: {
    kind: add (sum)
    operands:[10,6,6,10]
}

## Direction of transaction
Anchored to Actor 1. 
Result is always anchored to first actor.
### Addition
Sam had $100 and he [gets, adds, recieves] $12 from Mahesh
### Subtraction
Sam had $100 and he [gives, gifts, pays] $12 from Mahesh


### Script Goal
Output format for training and testing data 
 {
  "prompt": "John had 45$ and he got $10 from Mahesh",
  "operation": {
    "kind": add
    "operands":[45,10]
    }
 }

 Also ass option to the script --scale [1..100], default to 100
 Script should produce 100 (traning data) and 20 testing data.
 
 Total training rows 100,000
 Add scale factor and output only scale% rows randomly picked form generated rows.

 Also add a unit test in the script on 0.5% of the data.

 save training and testing data in json file __train.json __test.json 

### Ultimate Goal
Generate a crisp, clear and exact prompt.
