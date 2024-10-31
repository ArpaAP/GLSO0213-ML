import torch
import torch.nn as nn
from PIL import Image

def recog():
   path = "E:\\창의융합설계\\images\\"
   objects = ["새", "자동차", "물고기", "개", "고양이"]

   try:
      model = torch.load("trained.wgt")
   except Exception as e:
      print("학습 결과가 없습니다.")
      exit()

   device = 'cpu' # 'cpu'로 계산할지 'gpu'로 계산할지 지정, 'gpu'를 사용하려면 pytorch의 gpu 버전을 설치해야 함
   fp = open("res_0반_0팀.txt", "w")

   for i in range(1, 101):
      fileName = path + str(i).rjust(3, '0') + ".jpg"
      print(fileName)
      img = Image.open(fileName)
      if img.width != 128 or img.height != 128:
         print("파일 크기가 다릅니다. : ", fileName)
         continue
      line = []
      for y in range(128):
         for x in range(128):
            (r, g, b) = img.getpixel((x, y))
            line.append((r+g+b)) # 이 부분을 수정해야 함

      X = torch.FloatTensor(line).to(device) # 입력 데이터
      hypothesis = model(X) # foreward
      out = list(hypothesis.detach().cpu().numpy())
      print(fileName[-7:], out, objects[out.index(max(out))])
      outStr = str(i).rjust(3, '0') + " : " + objects[out.index(max(out))] + "\n"
      fp.write(outStr)
   fp.close()
   return

if __name__ == "__main__":
   recog()


