# Компилятор CUDA
NVCC = nvcc



# Исходный файл
SOURCE = kernel.cu

# Имя выходного исполняемого файла
TARGET = kernel.exe

# Правило по умолчанию
all: $(TARGET)

# Сборка исполняемого файла
$(TARGET): $(SOURCE)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SOURCE)

# Очистка собранных файлов
clean:
	del $(TARGET) *.obj *.exp *.lib

# Псевдоцель
.PHONY: all clean