import numpy as np

def probabilistic_binary_conversion(original_list):
    median_value = np.median(original_list)

    print(median_value)
    
    binary_list = []
    
    for value in original_list:

        probability = min(1, value / median_value)
        binary_value = np.random.binomial(1, probability)
        binary_list.append(binary_value)
    
    return binary_list

L = [10, 50, 100, 5, 75, 0]
result = probabilistic_binary_conversion(L)
print(result)