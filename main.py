import cv2
import numpy as np

img = cv2.imread("Don't worry.jpg")  # считываем заданное изображение, заблаговременно помещённое в файл

print(img.shape)  # выводим в консоль размеры изображения для понимания примерных размеров его отдельных областей
# получили размерность (727, 736, 3)

# Создаём базовую матрицу с размерностью исходного изображения с целью помещения в неё полученного решения:
new_img = np.zeros(img.shape, dtype='uint8')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # преобразуем цветное изображение в оттенки серого

img = cv2.GaussianBlur(img, (11, 11), 0)  # выполняем "размытие" картинки для последующей обработки

img = cv2.Canny(img, 10, 20)  # получаем чёрно-белое изображение, пригодное для поиска контуров

# Подготавливаем три однослойных "чёрных" холста с размерами, идентичными исходному изображению.
# В дальнейшем они будут использоваться для вычерчивания и "заливки" областей, соответствующих расположению
# каждой из строк исходного изображения. После этого они будут использоваться в качестве маски для побитовых операций.
area_1 = np.zeros(img.shape[:2], dtype='uint8')
area_2 = np.zeros(img.shape[:2], dtype='uint8')
area_3 = np.zeros(img.shape[:2], dtype='uint8')

area_1 = cv2.rectangle(area_1, (0, 0), (img.shape[1], 150), 255, -1)
area_1 = cv2.rectangle(area_1, (0, 150), (500, 170), 255, -1)
area_1 = cv2.rectangle(area_1, (500, 150), (550, 160), 255, -1)
area_1 = cv2.rectangle(area_1, (550, 150), (575, 156), 255, -1)

line_1 = cv2.bitwise_and(img, img, mask=area_1)  # получили отдельным изображением первую строку

area_2 = cv2.rectangle(area_2, (0, 150), (img.shape[1], 301), 255, -1)
area_2 = cv2.rectangle(area_2, (650, 301), (img.shape[1], 315), 255, -1)
area_2 = cv2.bitwise_or(area_1, area_2)  # объединяем области первой и второй строк,
area_2 = area_2 - area_1  # чтобы следующей операцией исключить всё, что относилось к первой строке

line_2 = cv2.bitwise_and(img, img, mask=area_2)  # получили отдельным изображением вторую строку

area_3 = cv2.rectangle(area_3, (0, 300), (img.shape[1], 450), 255, -1)
area_3 = cv2.bitwise_or(area_2, area_3)  # для областей второй и третьей строк производим аналогичные операции,
area_3 = area_3 - area_2  # чтобы выделить область, относящуюся исключительно к третьей строке

line_3 = cv2.bitwise_and(img, img, mask=area_3)  # получили отдельным изображением третью строку

# Теперь выполняем поиск контуров по каждой строке в отдельности:
con_1, hir_1 = cv2.findContours(line_1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
con_2, hir_2 = cv2.findContours(line_2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
con_3, hir_3 = cv2.findContours(line_3, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Воспроизводим окрашенные в разные цвета контуры в новом изображении:
cv2.drawContours(new_img, con_1, -1, (0, 255, 0), 1)
cv2.drawContours(new_img, con_2, -1, (0, 0, 255), 1)
cv2.drawContours(new_img, con_3, -1, (255, 0, 0), 1)

cv2.imshow("Don't worry! Be happy now!", new_img)  # выводим на экран полученное решение
cv2.waitKey(0)
