if __name__ == "__main__":    

    labels = "data/characters.csv"

    with open(labels, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        lines.pop(0)
        for lines in lines:
            line = lines.strip().split(',')
            hanzi = line[0]
            with open('data/labels.txt', 'a', encoding='utf-8') as f:
                f.write(hanzi + '\n')
            with open('data/piyin.txt', 'a', encoding='utf-8') as f:
                f.write(line[1] + '\n')
    print("Labels and Pinyin files created successfully.")
