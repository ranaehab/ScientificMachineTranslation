from collections import defaultdict

def find_duplicate_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]

    print("Sentences read from file:\n")
    for idx, sentence in enumerate(sentences, start=1):
        print(f"Line {idx}: {sentence}")
    
    print("\nChecking for duplicates...\n")

    sentence_map = defaultdict(list)
    duplicates_found = False

    # Map each sentence to its line numbers
    for idx, sentence in enumerate(sentences):
        sentence_map[sentence].append(idx + 1)

    # Check and print duplicates
    for sentence, lines in sentence_map.items():
        if len(lines) > 1:
            duplicates_found = True
            print(f"Duplicate sentence: \"{sentence}\"")
            print(f"Found on lines: {lines}\n")
    
    if not duplicates_found:
        print("Good")

# Example usage
if __name__ == "__main__":
    find_duplicate_sentences("D:/PHD Papers/scientific translation/GPT/ForGithub/Dataet/New Text Document.txt")
