import torch

# # Define tensors
# a1 = torch.randn(2, 1, 1, 15)
# b = torch.randn(15, 15)
#
# # Compute c for the first shape of a
# c1 = a1 * b
#
# b2 = b.unsqueeze(0)
# c2 = a1 * b2
#
# print(torch.allclose(c1, c2))  # Should print: True
#
# # Change the shape of a
# a2 = a1.permute(0, 1, 3, 2)
#
# # Compute c for the second shape of a
# c2 = a2 * b
#
# # Check if both results are the same
# print(torch.allclose(c1, c2))  # Should print: True

print(torch.tril(torch.ones(3,3)))
