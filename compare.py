with open("002반_02팀.txt", encoding="euc-kr") as f1:
    lines = f1.readlines()

with open("answer.txt", encoding="euc-kr") as f2:
    answer = f2.readlines()

correct = 0
total = 0

for i in range(len(lines)):
    if lines[i] == answer[i]:
        correct += 1
    else:
        print(f"line {i + 1}:")
        print("예측:", lines[i].strip())
        print("정답:", answer[i].strip())
    total += 1

print("정답률: %.2f%%" % (correct / total * 100))