Please make a prompt - want make a python script to make training and testing data. Script should have lot of useless data, script should capture final intent. if there are 2 operands an intent is subtract make it a - b, in all other case intent is addition (sum: a..z)
there are male and female actors with the name hindu, muslim, Christian. Please pick real commonly used names 10 each from each faith. male - can be father, friend, son, teacher brother, boyfriend. Female can be friend, girlfriend mother wife sister teacher. Avoid interfaith pairs
objects can be candies, cake, books.

Verbs can be gain, receive, add, get, obtain, find, == add. Invent your own words for subtract and write a prompt to make python file to make 100K training data and 20K test data. also a module which can detect the intent, Inent can also be +, - add subtract beside common verbs

Always first Subject any one (Amar, Akbar or Anthony) start with a positive quantity 
Example 

Amar had 3 marbles, then abkar and anthony gives 3 and 4 marbles to Amat (Sum Case)
Final output is relative to first subject - Amar in this case

Subtract Case
John had 15$ and he gave tim 3$
Final result relatibe to john

Add Case
Radha had 34 rupees, dad saurabh gave her 16 rupees
Final result relative to Radha

script should finally create each row with following structure
prompt - string
operands = int array [34, 39]
if there are exactly 2 oprands and intent subtract then it is subtract
if total oparands are not 2, then it is add by default
90% of statements should become 2 operands
3% should become 1 operands - like mahesh have 3$
7% should become 3+ operands like amar, akbar anthony have 3$ each in the pocket

Scaling: 
To facilitate manual inspection we need to create less data. So please add a command line flag --scale {1..100} meaning system must pick onlt scale% of random records in training and testing json records. scale default is 100%


