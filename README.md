# cuda_lab3
## Задание
Реализовать программу для накладывания фильтров на изображения. Возможные фильтры: размытие, выделение границ, избавление от шума. Изменить время
Для работы с графическими файлами рекомендуется использовать libpng (man libpng). Примеры использования библиотеки в /usr/share/doc/libpng12-dev/examples/
Считать изображение из файла, преобразовать в массив, отправить на CUDA Device, провести на устройстве обработку фильтром, вернуть изображение в память хоста, преобразовать в картинку, сохранить файл.
Сделать обработку изображения на 2 устройствах


## Описание программы
Для параллельности использовал потоки thread C++ 
Для считывания и записи изображения использовал библиотеку stb_image, так как ее настройка оказалась наиболее легкой (просто скопировать файлы .h в папку с программой). Реализовал программу только для 3 канальных изображений (RGB).
Размер блока выбрал 16х16 (256 потоков) так как такое число не очень большое (доступно на большинстве видеокарт) и делится на 32 - число одновременно обрабатывающихся потоков. 

### Основная идея:
Обработка изображения представляет собой применение свертки с ядром 3х3, следовательно для обработки каждого пикселя необхолдима информация из текущего пикселя а также соседних пикселей (halo) вокруг текущего (всего информация из 9 пикселей считая текущий) а также ядро светрки. Каждый из потоков обрабатывает свой пиксель (так как сетка содиржит в общем случае больше потоков чем пикселей в изображении, то не все потоки будут выполнять работу). По условию задания необходимо использовать общую память, она доступна всем потокам в блоке, и так как необходима информация о соседних пикселях размер общей памяти вычисляется как размеры блока по x и по y увеличенные на 2: по одному пикселю с каждой стороны и умноженные на число каналов изображения (3). В ядрах происходит вначале заполнение shared memory а затем после барьерной функции (чтобы дождаться когда вся общая память будет заполнена) происходит применение свертки.
Также по условию задания производится замер времени: общего, передачи данных на устройство, выполнения ядра, передачи памяти на хост. 

### Общий план работы программы:
1. Считывание изображения
2. Разбиение изображения на 2 части
3. 
4. Обработка изображения на двух устройствах (каждая часть в своем потоке):
    4.0 Инициализация памяти на устрйствах
    1.1 Передача изображения на устройство
    4.2 Заполнение общей памяти
    4.3 Применение свертки
    4.4 Передача обработанного изображения на хост
5. Соединение двух частей изображения
6. Сохранение изображения

### Реализованные фильтры:
- Blur - равномерный фильтр размытия
- Sobel - фильтр собеля для выделения границ


## Результат выполнения программы 
Программу запускал на 2 изображениях: большом (5824х3264) и маленьком (196х257).
Для запуска программы использовал менеджер очередей SLURM. Чтобы запустить его написал скрипт run.sbatch в котором загружается необходимые программы с помощью команды module load nvidia/cuda, затем вывожу информацию о устройствах GPU и запускаю программу с различными параметрами.

### Вывод в файл lab3.110334.out (старый)
```
=== Running lab3 program ===
SMALL IMAGE: blur
Loaded input_small.png: 196x257 channels=3
Saved output_small_blur.png
1)Host-to-Device copy time (ms): 4.2944e-02
1)Kernel time (ms): 2.3262e+01
1)Device-to-Host copy time (ms): 6.4544e-02
2)Host-to-Device copy time (ms): 3.9360e-02
2)Kernel time (ms): 4.7153e+00
2)Device-to-Host copy time (ms): 7.6768e-02
Total time (ms): 5.4465e+02
SMALL IMAGE: sobel
Loaded input_small.png: 196x257 channels=3
Saved output_small_sobel.png
1)Host-to-Device copy time (ms): 4.5952e-02
1)Kernel time (ms): 1.1401e+01
1)Device-to-Host copy time (ms): 7.2736e-02
2)Host-to-Device copy time (ms): 4.1088e-02
2)Kernel time (ms): 4.7029e+00
2)Device-to-Host copy time (ms): 7.8592e-02
Total time (ms): 3.1063e+02
BIG IMAGE: blur
Loaded input.png: 5824x3264 channels=3
Saved output_blur.png
1)Host-to-Device copy time (ms): 7.2407e+00
1)Kernel time (ms): 1.2230e+01
1)Device-to-Host copy time (ms): 1.1496e+01
2)Host-to-Device copy time (ms): 7.2262e+00
2)Kernel time (ms): 2.8893e+00
2)Device-to-Host copy time (ms): 1.2254e+01
Total time (ms): 9.9556e+03
BIG IMAGE: sobel
Loaded input.png: 5824x3264 channels=3
Saved output_sobel.png
1)Host-to-Device copy time (ms): 7.2739e+00
1)Kernel time (ms): 7.8328e+00
1)Device-to-Host copy time (ms): 1.1460e+01
2)Host-to-Device copy time (ms): 7.2387e+00
2)Kernel time (ms): 2.8808e+00
2)Device-to-Host copy time (ms): 1.2154e+01
Total time (ms): 1.0066e+04
Job completed successfully!
```
|



### Вывод в файл lab3.113122.out (новый)
```
=== Running lab3 program ===
SMALL IMAGE: blur
Loaded input_small.png: 196x257 channels=3
GPU 0: Allocation time (ms): 0.144384
GPU 0: H2D memcpy time (ms): 0.062176
GPU 1: Allocation time (ms): 0.003072
GPU 1: H2D memcpy time (ms): 0.041216
GPU 0: Kernel time (ms): 584.997314
GPU 0: D2H memcpy time (ms): 0.083616
GPU 1: Kernel time (ms): 563.282959
GPU 1: D2H memcpy time (ms): 0.073472
Saved output_small_blur.png
Total time (ms): 850.396
SMALL IMAGE: sobel
Loaded input_small.png: 196x257 channels=3
GPU 0: Allocation time (ms): 0.120832
GPU 0: H2D memcpy time (ms): 0.052416
GPU 0: Kernel time (ms): 30.791679
GPU 0: D2H memcpy time (ms): 0.088512
GPU 1: Allocation time (ms): 0.003072
GPU 1: H2D memcpy time (ms): 0.041024
GPU 1: Kernel time (ms): 4.702912
GPU 1: D2H memcpy time (ms): 0.077248
Saved output_small_sobel.png
Total time (ms): 268.468
BIG IMAGE: blur
Loaded input.png: 5824x3264 channels=3
GPU 0: Allocation time (ms): 0.221184
GPU 0: H2D memcpy time (ms): 6.491648
GPU 0: Kernel time (ms): 40.994846
GPU 1: Allocation time (ms): 0.003072
GPU 1: H2D memcpy time (ms): 6.454784
GPU 1: Kernel time (ms): 2.873248
GPU 0: D2H memcpy time (ms): 19.224672
GPU 1: D2H memcpy time (ms): 19.074688
Saved output_blur.png
Total time (ms): 9968.28
BIG IMAGE: sobel
Loaded input.png: 5824x3264 channels=3
GPU 0: Allocation time (ms): 0.241632
GPU 0: H2D memcpy time (ms): 6.563648
GPU 0: Kernel time (ms): 26.804192
GPU 1: Allocation time (ms): 0.003072
GPU 1: H2D memcpy time (ms): 6.537696
GPU 1: Kernel time (ms): 2.674432
GPU 0: D2H memcpy time (ms): 19.103807
GPU 1: D2H memcpy time (ms): 19.193184
Saved output_sobel.png
Total time (ms): 10214
Job completed successfully!
```


## Вывод
При выполнении вычислений на 2 устройствах в двух параллельных потоках общее время обработки на 2 порядка меньше. 
