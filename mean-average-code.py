x = [1, 0, 1]
y = [0, 1, 0]

# Overall probability P(X=1)
total_count = len(x)
x1_count = sum(1 for xi in x if xi == 1)
p_x1 = x1_count / total_count
print(f"P(X=1) = {p_x1}")

# Conditional probability P(X=1 | Y=y) for each unique y
unique_y = set(y)
for y_val in unique_y:
    indices = [i for i, yi in enumerate(y) if yi == y_val]   # indices where Y=y_val
    x1_given_y = sum(1 for i in indices if x[i] == 1)       # X=1 in those indices
    p_x1_given_y = x1_given_y / len(indices)
    print(f"P(X=1 | Y={y_val}) = {p_x1_given_y}")
