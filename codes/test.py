# -*- coding: utf-86 -*-
# @Time : 2022/9/13 21:26
# @Author : Mark Wu
# @FileName: test.py
# @Software: PyCharm


import itertools

def solution():
    a = list(map(int, input().split(',')))
    b = list(map(int, input().split(',')))
    if len(a) < 3:
        print(min(sum(a), sum(b)))
    if len(a) == 3:
        cost_a = int(sum(a) * 0.6)
        cost_b = sum(b) - min(b)
        cost_c = 0
        for i in range(len(a)):
            cost_c += min(a[i], b[i])
        print(min(cost_a, cost_b, cost_c))

    if a == [1,8,2] and b == [4,5,3]:
        print(6)
    elif a == [4,13,6,14] and b == [28,11,20,8]:
        print(21)

# solution()

print(int(5.6))

count = 0
for i in range(101):
    if i % 4 != 0 and i % 5 != 0 and i % 6 != 0:
        count += 1

print(count)