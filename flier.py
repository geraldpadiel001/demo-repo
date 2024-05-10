
import csv

def create_csv(filename):
    # Define column names
    fields = ['Image', 'Label', 'Name']

    # Create or open the CSV file in write mode
    with open(filename, 'w', newline='') as csvfile:
        # Create a CSV writer object
        csvwriter = csv.writer(csvfile)

        # Write the column names to the CSV file
        csvwriter.writerow(fields)

def add_data_to_csv(filename, data):
    # Open the CSV file in append mode
    with open(filename, 'a', newline='') as csvfile:
        # Create a CSV writer object
        csvwriter = csv.writer(csvfile)

        # Write the data to the CSV file
        csvwriter.writerow(data)

# Example usage:
csv_filename = 'data_6.csv'

# Create the CSV file with column names
create_csv(csv_filename)

# Add data to the CSV file



import os

def list_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"{file}")
            data_to_add = [file_path, 'illegal', file.split(" ")[0]]
            add_data_to_csv(csv_filename, data_to_add)

# Replace 'path_to_folder' with the path to your folder
folder_path = 'training_data'

# Call the function to list files and their paths
list_files(folder_path)
