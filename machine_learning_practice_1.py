# import numpy as np
# import matplotlib.pyplot as plt
# import math

# theta = float(input("초기 각도값(degree) 입력 ㄱㄱ ")) # degree
# print("theta : ",theta)
# initial_velociy = float(input("초기 속도를 입력 ㄱㄱ ")) # m/s
# initial_height = float(input("초기 높이를 입력 ㄱ "))
# gravity_constant = 9.81 # 중력 가속도


# t = np.arange(0, 100, 0.1)

# x = initial_velociy * np.cos(np.deg2rad(theta))*t # 1차원 리스트 -> t때문에 ㅇㅇ
# y = initial_height + initial_velociy * np.sin(np.deg2rad(theta))*t - gravity_constant*(t**2)/2 # 1차원 리스트



# # print(x)
# # print(y)

# #print(x[np.where(y >= 0)[0][-1]])

# index = np.where(y >= 0)[0][-1]

# ground_time = t[index+1]
# print("땅에 떨어지는 시간 : ", ground_time)

# x_ground = x[index+1]
# print("땅에 떨어지는 지점의 x좌표 : ", x_ground)

# y_max_index = np.argmax(y)
# y_max = y[y_max_index]

# print("최대 높이 : ", y_max)


# plt.plot(x[:index+1], y[:index+1], 'ro', label="trajectory")
# plt.title("parabolic graph")
# plt.xlabel("distance(m)")
# plt.ylabel("height(m)")
# plt.grid()
# plt.legend()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

degree = float(input("초기 각도값(degree) 입력 ㄱㄱ "))
initial_velocity = float(input("초기 속도를 입력 ㄱㄱ "))
initial_height = float(input("초기 높이를 입력 ㄱ "))

vx = initial_velocity * np.cos(np.deg2rad(degree))
vy = initial_velocity * np.sin(np.deg2rad(degree))

g = 9.81
t = (-vy - np.sqrt(vy**2 + 2*g*initial_height)) / (-g)
x = vx * t
y = initial_height + vy * t - 0.5 * g * t**2

print("땅에 떨어지는 시간 : ", t)
print("땅에 떨어지는 지점의 x좌표 : ", x)
print("최대 높이 : ", initial_height + (vy**2) / (2*g))

t_values = np.linspace(0, t, num=100)
x_values = vx * t_values
y_values = initial_height + vy * t_values - 0.5 * g * t_values**2

plt.plot(x_values, y_values, 'ro', label="trajectory")
plt.title("parabolic graph")
plt.xlabel("distance(m)")
plt.ylabel("height(m)")
plt.grid()
plt.legend()
plt.show()