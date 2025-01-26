# Input file path
filename = "rachel_new_data_copy.txt"  # Change to your input file path

# Open the file to read and write
with open(filename, 'r+') as file:
    # Read all lines from the file
    lines = file.readlines()
    
    # Go to the beginning of the file for overwriting
    file.seek(0)
    
    # Iterate over each line in the file
    for line in lines:
        # Split the line by commas
        values = line.strip().split(' ')
        
        # Parse time step, num1, num2
        time_step = values[0].strip()
        num1 = float(values[1].strip())
        num2 = float(values[2].strip())
        
        # Calculate num1 - num2
        diff = num1 - num2
        
        # Write the new format: time step, num1, num2, num1 - num2
        file.write(f"{time_step}, {num1}, {num2}, {diff}\n")

    # Truncate the file to remove any old content beyond the updated lines (in case the new data is shorter)
    file.truncate()

print("File has been modified.")
