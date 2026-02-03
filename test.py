over = 2.4
whole_num = int(over)
print(whole_num)
fractional_num = over - whole_num
print(fractional_num)
print(round(fractional_num * 10))
balls_faced = whole_num * 6 + round(fractional_num * 10)
print(balls_faced) 