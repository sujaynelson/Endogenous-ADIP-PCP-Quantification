import numpy as np

# Replace 'path_to_file.npy' with your actual file path
mask_path = "all_data_structured/myexperiment/masks/mask_div.npy"
data = np.load(mask_path, allow_pickle=True)
print(data)
print("Shape:", data.shape)
print("Dtype:", data.dtype)

# Analyze the loaded object
print("Type:", type(data))
if isinstance(data, dict):
    print("Keys:", data.keys())
elif isinstance(data, list):
    print("Length:", len(data))
    print("First item type:", type(data[0]))
else:
    print("Value:", data)

# Print shape, dtype, and ndim
print("Shape:", data.shape)
print("Dtype:", data.dtype)
print("Ndim:", data.ndim)

# Extract the dictionary from the 0-dimensional array
obj = data.item()
print("Extracted object type:", type(obj))
if isinstance(obj, dict):
    print("Keys:", obj.keys())
    # Loop through keys and print values
    for key in obj:
        print(f"Key: {key}")
        print(f"Value type: {type(obj[key])}")
        print(f"Value: {obj[key]}")

# Access specific fields
print("masks:", obj["masks"])
print("filename:", obj["filename"])
print("cellprob_threshold:", obj["cellprob_threshold"])

# Extract and inspect masks array
obj = data.item()
if isinstance(obj, dict):
    if "masks" in obj:
        masks_array = obj["masks"]
        print("Masks array shape:", masks_array.shape)
        print("Masks array dtype:", masks_array.dtype)
        print("First mask:", masks_array[0])
    else:
        print("'masks' key not found in the dictionary.")

# Save masks to a separate file
obj = data.item()
if isinstance(obj, dict):
    if "masks" in obj:
        masks_array = obj["masks"]
        np.save("masks_extracted.npy", masks_array)
        print("Masks array saved as masks_extracted.npy")
