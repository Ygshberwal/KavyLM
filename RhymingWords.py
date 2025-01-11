import bisect

def reverse_word(word):
    return word[::-1]

def load_reversed_sorted_dict(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().splitlines()

def find_rhyming_words(input_word, reversed_sorted_dict, match_length=3):
    # Reverse the input word
    reversed_input = reverse_word(input_word)
    # Use only the first `match_length` characters for matching
    reversed_input_prefix = reversed_input[:match_length]

    # Binary search to find the starting position of matching words
    idx = bisect.bisect_left(reversed_sorted_dict, reversed_input_prefix)
    rhyming_words = []

    # Collect words that match the prefix
    while idx < len(reversed_sorted_dict) and reversed_sorted_dict[idx].startswith(reversed_input_prefix):
        rhyming_words.append(reverse_word(reversed_sorted_dict[idx]))
        idx += 1

    return rhyming_words

# Input and Dictionary File
reversed_dict_file = "reversed_sorted_dict.txt"  # Path to the pre-generated dictionary
reversed_sorted_dict = load_reversed_sorted_dict(reversed_dict_file)

# Find rhyming words
input_word = input("Enter a Hindi word: ")
match_length = int(input("Enter the number of characters to match (2 or 3): "))

if match_length not in [2, 3]:
    print("Invalid match length! Please enter 2 or 3.")
else:
    rhyming_words = find_rhyming_words(input_word, reversed_sorted_dict, match_length=match_length)

    # Print the results
    print(f"Rhyming words for '{input_word}' (last {match_length} characters match): {', '.join(rhyming_words) if rhyming_words else 'None found'}")
