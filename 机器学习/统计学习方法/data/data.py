# @Author: phd
# @Date: 19-3-28
# @Site: github.com/phdsky
# @Description:
#   Prepare binary classification mnist data
#   label(number) : -1(0 ~ 4)  1(5 ~ 9)

import os
import pandas as pd

mnist = pd.DataFrame(pd.read_csv("mnist.csv"))
print(mnist.shape)

# First convert 0 ~ 4 to -1
mnist.loc[mnist['label'] < 5, 'label'] = -1

# Then convert 5 to 9 to 1
mnist.loc[mnist['label'] >= 5, 'label'] = 1

file_name = "mnist_binary.csv"

if not os.path.exists(file_name):
    mnist.to_csv(file_name, index=False)
else:
    print("%s already exists" % file_name)
