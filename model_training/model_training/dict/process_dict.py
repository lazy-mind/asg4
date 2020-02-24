input_file_path = "glove.twitter.27B.25d.txt"
output_file_path = "glove_vector.txt"

content = ["<pad> 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n"]

file = open(input_file_path, "r")
for word_vector in file:
    if word_vector.split()[0] == "<unk>":
        a = word_vector.split()[1:]
        a.insert(0,"<unknown>")
        new_row = " ".join(a)
        content.insert(1, new_row+"\n")
    else:
        content.append(word_vector)
file.close()

print(len(content))

file = open(output_file_path, "a")
for word_vector in content:
    file.write(word_vector)
file.close()