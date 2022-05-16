import pandas as pd
import numpy as np

df1 = pd.DataFrame(
    np.random.randint(25, size=(4, 4)),
    index=["1", "2", "3", "4"],
    columns=["A", "B", "C", "D"],
)

df2 = pd.DataFrame(
    np.random.randint(25, size=(6, 4)),
    index=["5", "6", "7", "8", "9", "10"],
    columns=["A", "B", "C", "D"],
)

df3 = pd.DataFrame(np.random.randint(25, size=(4, 4)), columns=["A", "B", "C", "D"])

df4 = pd.DataFrame(np.random.randint(25, size=(4, 4)), columns=["E", "F", "G", "H"])

print(df1)
print("\n")
print(df2)
print("\n")
print(df3)
print("\n")
print(df4)
print("\n")


# concatenating df1 and df2 along rows
vertical_concat = pd.concat([df1, df2], axis=0)

# concatenating df3 and df4 along columns
horizontal_concat = pd.concat([df3, df4], axis=1)

print(vertical_concat)
print("\n")
print(horizontal_concat)
print("\n")
