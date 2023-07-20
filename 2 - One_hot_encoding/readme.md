Categorical data, especially nominal data, cannot be directly replaced with numbers meaningfully, 
as it can introduce false ordinal relationships between the categories. For example, 
suppose you encode categories as numbers 1, 2, 3, and 4. In that case, the model might falsely interpret 
that category 4 is somehow "greater" or "better" than category 1, which is not the case for nominal data.

To handle categorical data appropriately, one hot encoding is commonly used.
One hot encoding is converting categorical variables into binary vectors where each category 
is represented as a unique binary value (0 or 1). Each category is treated as a separate binary feature,
and only one element in the binary vector is 1 (hot), while the others are 0 (cold).
