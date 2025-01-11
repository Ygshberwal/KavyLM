# generate_dict.py

def reverse_word(word):
    return word[::-1]

def create_reversed_sorted_dict(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        words = set()  # Use a set to eliminate duplicates
        for line in file:
            line_words = line.strip().split()  # Split words by spaces
            for word in line_words:
                # Include only Hindi words based on Unicode range
                if all('\u0900' <= char <= '\u097F' for char in word):
                    words.add(word)  # Add unique words to the set

    # Create a reversed sorted list
    reversed_words = sorted(reverse_word(word) for word in words)

    # Save the reversed sorted dictionary to a file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write("\n".join(reversed_words))

    print(f"Reversed sorted dictionary created and saved to '{output_file}'.")

# Input and Output File Paths
input_file = "hindi_words.txt"  # Replace with your input file
output_file = "reversed_sorted_dict.txt"

# Generate the reversed sorted dictionary
create_reversed_sorted_dict(input_file, output_file)
