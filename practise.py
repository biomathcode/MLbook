
a = 10+3j
b = 9-1j
print(abs(a-b))
str ='''This is a multiline string that can continue on another line
presseing the enter key'''
print(str)

s = 'pratik sharma'
#    0123456789012
print(s[0])
print(s[2])
print(s[-1])
print(s[10])
print(s[0:13])

a = 'a'
b ='b'
c = a + ' ' + b
print(c)

str1 = input("Input the string:")
def add_strint(str1):
    length = len(str1)
    if length >1:
        if str[-3:] =='ing':
            str1+='ly'
        else:
            str1+='ing'
    return str1
print(add_strint(str1))
# The great thing about python is that you can assign variable simultaneoulsy
a, b = 3, 4
def string_length(str1):
    count = 0
    for char in str1:
        count += 1
    return count
print(string_length('Pratik Sharma'))
print('Person_name')
Person_name = input()
def char_frequency(str1):
    dict ={}
    for n in str1:
        keys = dict.keys()
        if n in keys:
            dict[n] += 1
        else:
            dict[n] = 1
    return dict
print(char_frequency(Person_name))
def last_first(str1):
    if len(str1) <2:
        return ''
    return str[0:2] + str[-2:]

print(last_first(Person_name))

var1 = [2,3,4,5,6]
var2 = (2,3,4,5,6)
# var1 is called a list and var2 is called a tuple
a =[1,2,3]
b = [2,3,4]
print(c)
# iteration of data
# The capital is india is New Delhi
# The capital of france is Paris
# The Capital of Pakistan is Lahore
# 200 countries
# what will you do
data = {
    'France': 'Paris',
    'India': 'New Delhi',
    'Italy':'Rome',
    'Germany':'Berlin',
    'Pakistan':'Lahore',
    'USA':'Washington DC',
}
for country in data:
    print('The capital of ' + country + ' is ' + data[country])
s = 0
def unability(s):
    for i in range(20):
        print(s +i)
    while s < 10:
        print("It is up to 10")
        s += 1
unability(s)



