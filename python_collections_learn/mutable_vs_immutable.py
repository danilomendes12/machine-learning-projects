# tuple is an immutable type
def test_for_tuple():
    x = (1, 2, 3)
    y = x  # a new reference to a tuple object is created

    x = (1, 2)  # a new tuple object is created, but it doesn't change y
    print(y)  # (1, 2, 3)


# list is a mutable type
def test_for_list():
    x = [1, 2, 3]
    y = x  # y is a new reference to the list object x

    x[0] = 0  # x is mutable, so it changes y
    print(y)  # [0, 2, 3]


print("Test for tuple:")
test_for_tuple()

print("Test for list:")
test_for_list()
#%%
