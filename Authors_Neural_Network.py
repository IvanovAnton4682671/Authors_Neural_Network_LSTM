
#Так, ну что, пятилетку за два дня, как говорится)
#Цель - реализовать нейронную сеть, которая будет принимать текст и выбранного автора, анализировать его, и возвращать исходный текст,
#но переписанный в стиле выбранного автора
#Для удобства использования напишем GUI на PyQt или Tkinter
#Саму сеть будем писать с использованием библиотеки PyTorch
#Поскольку сеть будет обрабатывать естественный язык (с учётом контекста), то наиболее подходящая модель - RNN, а именно LSTM
#Для обучения сети нужно будет провести обработку обучающего материала (токенизацию и векторизацию)
#Входной слой будет представлять собой эмбеддинги слов, которые будут пропускаться через LSTM-слои, а на выходе будет softmax-слой
#В качестве функции потерь будем использовать кросс-энтропию, а в качестве оптимизации - стохастический градиентный спуск или Adam

#Скаченные библиотеки для работы: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118, pip install torchtext

############################## ИМПОРТИРУЕМЫЕ БИБЛИОТЕКИ ##############################

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchtext
import torchtext.vocab

import tkinter as tk
from tkinter import ttk
import string
import threading

############################## ОСНОВНОЙ КЛАСС НЕЙРОСЕТИ ##############################

#наследуемся от базового класса для всех модулей нейронной сети
class LSTM_Network(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, output_size: int, device: torch.device) -> None:
        """
        Функция, которая инициализирует нейронную сеть
        - Args:
          - self - объект-сеть
          - input_size - размер входного слоя
          - hidden_size - размер скрытого состояния LSTM
          - output_size - размер выходного слоя
          - device - устройство-обработчик вычислений
        - Return:
          - отсутствует
        """
        #вызываем базовый конструктор
        super(LSTM_Network, self).__init__()
        #создаём скрытый слой
        self.hidden_size = hidden_size
        #создаём слой эмбеддингов слов, где input_size - размер словаря (количество уникальных слов),
        #а hidden_size - размерность эмбеддинга, и пробуем перенести вычисления на GPU
        self.embedding = nn.Embedding(input_size, hidden_size).to(device)
        #создаём LSTM-слой, определяем его размерность входа и размерность скрытого состояния, которые оба равны hidden_size
        #и пробуем перенести вычисления на GPU
        self.lstm = nn.LSTM(hidden_size, hidden_size).to(device)
        #создаём полносвязный слой, который принимает на вход скрытое состояние LSTM
        #и преобразует его в выходной вектор размерности output_size, а ещё пробуем перенести вычисления на GPU
        self.fc = nn.Linear(hidden_size, output_size).to(device)
        #создаём softmax-слой, который применяется к выходному вектору для получения вероятностей классов
        #(используем логарифмическую функцию, чтобы избежать численной нестабильности)
        self.softmax = nn.LogSoftmax(dim=1)

    def trying_to_use_gpu(self, loss_function: torch.nn.CrossEntropyLoss, optimizer: torch.optim.SGD) -> Tuple["LSTM_Network", torch.nn.CrossEntropyLoss, torch.optim.SGD]:
        """
        Функция, которая пробует перенести основные вычисления на GPU
        - Args:
          - self - объект-сеть
          - loss_function - кросс-энтропийная функция потерь
          - optimizer - оптимизатор
        - Return:
        - новые сеть, функция потерь и оптимизатор
        """
        #добавление обработки на GPU для ускорения
        new_network = self.to(device)
        new_loss_function = loss_function.to(device)
        for param in optimizer.param_groups[0]['params']:
            param.data = param.data.to(device)
        return new_network, new_loss_function, optimizer

    def forward(self, input: torch.Tensor, hidden: tuple, device: torch.device) -> Tuple[torch.nn.LogSoftmax, torch.nn.LSTM]:
        """
        Функция, которая определяет прохождение данных через нейронную сеть
        - Args:
          - self - объект-сеть
          - input - представление входных данных
          - hidden - существующий скрытый слой
          - device - устройство-обработчик вычислений
        - Return:
          - новое представление обработанных входных данных
        """
        #проверяем, есть ли входные данные в словаре
        if torch.max(input) >= self.embedding.num_embeddings:
            raise ValueError("Input index out of range of embedding matrix.")
        #преобразуем входные данные в эмбеддинги
        #embedded = self.embedding(input).view(1, 1, -1)
        #добавляем размерность для последовательности и пробуем перенести вычисления на GPU
        embedded = self.embedding(input.to(device)).unsqueeze(1)
        #пробуем перенести hidden-слой на GPU
        hidden = tuple(h.to(device) for h in hidden)
        #теперь передаём их через lstm-слой и пробуем перенести вычисления на GPU
        output, hidden = self.lstm(embedded.to(device), hidden)
        #и через полносвязный слой
        #output = self.fc(output.view(1, -1))
        #удаляем измерение последовательности перед применением полносвязного слоя
        #и пробуем перенести вычисления на GPU
        output = self.fc(output.squeeze(1).to(device))
        #применяем softmax-функцию для получения вероятностей
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self) -> Tuple[torch.Tensor]:
        """
        Функция, которая инициализирует скрытое состояние lstm
        - Args:
          - self - объект-сеть
        - Return:
          - кортеж из двух тензоров, заполненных нулями
        """
        return (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))

    def build_vocab(self, input_text: str) -> Tuple[torchtext.vocab.Vocab, list[int]]:
        """
        Функция, которая составляет из входного текста словарь
        - Args:
          - self - объект-сеть
          - input_text - входной текст
        - Return:
          - словарь из исходного текста и список индексов
        """
        #разбиваем текст на слова (токенизация)
        tokens = input_text.split()
        #создаём объект-словарь, который автоматически преобразует слова в индексы
        vocab = torchtext.vocab.build_vocab_from_iterator([tokens], specials=['<unk>', '<pad>'], min_freq=1)
        #создаём список индексов
        indexed_data = [vocab[token] for token in tokens]
        #ДЛЯ ОТЛАДКИ
        #for word in vocab.get_itos():
            #print(f"Слово: {word}, Индекс: {vocab[word]}")
        #print("А теперь индексы.")
        #for index in indexed_data:
            #print(index)
        return vocab, indexed_data

    def prepare_data(self, data_file_path: str, batch_size: int, number_of_workers: int) -> DataLoader:
        """
        Функция, которая подготавливает данные для обучения нейронной сети
        - Args:
          - self - объект-сеть
          - data_file_path - путь к файлу с данными
          - batch_size - размер батча
          - number_of_workers - количество потоков обработки данных
        - Returns:
          - подготовленные данные, разделённые на батчи
        """
        with open(data_file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        _, indexed_data = self.build_vocab(text)
        #преобразуем список индексов в тензор
        data_tensor = torch.tensor(indexed_data, dtype=torch.long)
        #создаем DataLoader для удобства работы с батчами
        dataset = TensorDataset(data_tensor)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=number_of_workers)
        #ДЛЯ ОТЛАДКИ
        #for batch in data_loader:
            #print(batch)
        return data_loader

    def train(self, loss_function: torch.nn.CrossEntropyLoss, optimizer: torch.optim.SGD, data_loader: DataLoader, epochs: int, number_save: int, device: torch.device) -> None:
        """
        Функция, которая выполняет обучение нейронной сети
        - Args:
          - self - объект-сеть
          - loss_function - кросс-энтропийная функция потерь
          - optimizer - метод оптимизации (стохастический градиентный спуск)
          - data_loader - содержит обучающие данные
          - epochs - количество эпох обучения
          - number_save - каждую эту эпоху будет происходит сохранение весов
          - device - устройство-обработчик вычислений
        - Return:
          - отсутствует
        """
        #ДЛЯ ОТЛАДКИ
        all_info = []
        j = 0
        for epoch in range(epochs):
            total_loss = 0
            #ДЛЯ ОТЛАДКИ
            i = 0
            for batch in data_loader:
                #получаем входные данные из батча
                inputs = batch[0]
                #ДЛЯ ОТЛАДКИ
                min_index = torch.min(inputs)
                max_index = torch.max(inputs)
                #print(f"Текущая эпоха: {epoch + 1}")
                #print(f"Номер текущего батча: {i + 1}; всего батчей: {len(data_loader)}")
                #print(f"Минимальный индекс входных данных: {min_index}")
                #print(f"Максимальный индекс входных данных: {max_index}")
                #print(f"Размер словаря (количество уникальных слов): {self.embedding.num_embeddings}")
                print(f"Текущая эпоха: {epoch + 1}; номер текущего батча: {i + 1}, всего батчей: {len(data_loader)}; минимальный индекс: {min_index}, максимальный индекс: {max_index}; размер словаря: {self.embedding.num_embeddings}; общий прогресс обучения: {j + 1}/{len(data_loader) * epochs} ({(j / (len(data_loader) * epochs)) * 100}%)  ", end="\r")
                #в качестве целевых значений берём те же, что и входные данные (autoencoder)
                targets = inputs
                #пробуем перенести вычисления на GPU
                inputs, targets = inputs.to(device), targets.to(device)
                #обнуляем градиенты
                optimizer.zero_grad()
                #прямой проход
                hidden = self.init_hidden()
                outputs, _ = self.forward(inputs, hidden, device)
                #вычисление функции потерь
                loss = loss_function(outputs.squeeze(), targets)
                total_loss += loss.item()
                #обратный проход и обновление весов
                loss.backward()
                optimizer.step()
                i += 1
                j += 1
            info = f"Epoch {epoch + 1}, Loss: {total_loss / len(data_loader)}"
            print(info)
            all_info.append(info)
            if (epoch + 1) % number_save == 0:
                self.save_network_weights("APCh_weights_epoch_{}_v10.pth".format(epoch + 1))
        for info in all_info:
            print(info)

    def data_preparation_and_training(self, data_file_path: str, batch_size: int, number_of_workers: int, loss_function: torch.nn.CrossEntropyLoss, optimizer: torch.optim.SGD, epochs: int, number_save: int, device: torch.device) -> None:
        """
        Функция, которая подготавливает данные для обучения и обучает сеть
        - Args:
          - data_file_path - путь к файлу с данными
          - batch_size - размер батча
          - number_of_workers - количество параллельных потоков
          - loss_function - кросс-энтропийная функция потерь
          - optimizer - оптимизатор
          - epochs - количество эпох
          - number_save - каждую эту эпоху будет происходит сохранение весов
          - device - устройство-обработчик вычислений
        - Return:
          - отсутствует
        """
        #подготовленные данные для обучения
        data_loader = self.prepare_data(data_file_path, batch_size, number_of_workers)
        #обучение сети
        self.train(loss_function, optimizer, data_loader, epochs, number_save, device)

    def generate_new_text(self, initial_text: str, author: str, device: torch.device, max_length: int, temperature: float, label_progress: tk.Label) -> str:
        """
        Функция для генерации текста в стиле указанного литературного автора
        - Args:
          - self - объект-сеть
          - initial_text - исходный текст
          - author - автор, в стиле которого нужно сгенерировать новый текст
          - device - устройство-обработчик вычислений
          - max_length - максимальная длина нового текста
          - temperature - коэффициент разнообразия генерации
          - label_progress - метка-прогресс генерации
        - Returns:
          - новый текст на основе исходного
        """
        #переводим исходный текст в индексы токенов
        _, indexed_data = self.build_vocab(initial_text)
        input_tensor = torch.tensor(indexed_data, dtype=torch.long, device=device)
        #инициализируем скрытое состояние LSTM
        hidden = self.init_hidden()
        #анализ исходного текста
        with torch.no_grad():
            for _ in range(input_tensor.size(0) - 1):
                output, hidden = self.forward(input_tensor, hidden, device)
        #генерация нового текста в стиле выбранного автора
        generated_text = ""
        #generated_text = initial_text
        vocab, _ = self.build_vocab(initial_text)
        with torch.no_grad():
            i = 0
            for _ in range(max_length):
                #прямой проход LSTM модели
                #output, hidden = self.forward(input_tensor, hidden, device)
                output, hidden = self.forward(input_tensor[-1].unsqueeze(0), hidden, device) #используем только последний токен в качестве входа
                #применяем softmax с "температурой" для управления разнообразием генерации
                output_dist = F.softmax(output.view(-1) / temperature, dim=0)
                #выбираем следующий индекс токена на основе распределения вероятностей
                top_i = torch.multinomial(output_dist, 1)[0].item()
                #проверяем, что индекс находится в пределах длины словаря
                print(f"Прогресс генерации: {i + 1}/{max_length} ({((i + 1) / max_length) * 100}%)            ", end="\r")
                info = f"Прогресс генерации: {round(((i + 1) / max_length), 5) * 100}%"
                label_progress.config(text=info)
                i += 1
                if 0 <= top_i < len(vocab):
                    #print(f"Токен, который входит в словарь: {top_i}")
                    generated_text += " " + vocab.get_itos()[top_i]
                    if vocab.get_itos()[top_i] == ".":
                        break
                    #записываем выбранный индекс в тензор
                    top_i_tensor = torch.tensor(top_i, device=device).unsqueeze(0)
                    input_tensor = torch.cat((input_tensor, top_i_tensor), dim=0)
                else:
                    #если индекс находится за пределами, игнорируем его (параметр end="\r" заменяет строку, а не дописывает новую)
                    #print(f"Индекс {top_i} выходит за пределы длины словаря")
                    continue
        return generated_text

    def network_testing(self, number_of_results: int, input_text: str, author: str, device: torch.device, max_length: int, temperature: float, insert_result: callable, output_textarea: tk.Text, label_progress: tk.Label) -> list[str]:
        """
        Функция, которая тестирует работу нейросети
        - Args:
          - self - объект-сеть
          - number_of_results - количество результатов
          - input_text - входной текст
          - author - автор, в стиле которого нужно переписать текст
          - device - устройство-обработчик вычислений
          - max_length - максимальная длина ответа
          - temperature - коэффициент разнообразия генерации
          - insert_result - функция обратного вызова (нужно для корректной работы в параллельном потоке)
          - output_textarea - зона вывода результата
          - label_progress - метка-прогресс генерации
        - Return:
          - список ответов нейросети
        """
        all_results = []
        i = 0
        for _ in range(number_of_results):
            #print(f"Генерируем {i + 1} вариант:")
            #генерируем новый текст на основе исходного
            result = self.generate_new_text(input_text, author, device, max_length, temperature, label_progress)
            all_results.append(result)
            i += 1
        #return all_results
        #перенаправление результата в другую функцию
        if number_of_results == 1:
          insert_result(all_results[0], output_textarea)

    def print_results(self, results_before: list[str], results_after: list[str]) -> None:
        """
        Функция, которая печатает результаты работы сети
        - Args:
          - self - объект-сеть
          - results_before - результаты до обучения
          - results_after - результаты после обучения
        - Return:
          - отсутствует
        """
        print("Результаты до обучения")
        for result in results_before:
            print(result)
        print("Результаты после обучения")
        for result in results_after:
            print(result)

    def save_network_weights(self, file_path: str) -> None:
        """
        Функция, которая сохраняет веса нейронной сети
        - Args:
          - self - объект-сеть
          - file_path - путь к файлу
        - Return:
          - отсутствует
        """
        torch.save(self.state_dict(), file_path)
        print(f"Веса успешно сохранились по пути {file_path}")

    def load_network_weights(self, file_path: str) -> None:
        """
        Функция, которая загружает веса нейронной сети
        - Args:
          - self - объект-сеть
          - file_path - путь к файлу
        - Return:
          - отсутствует
        """
        self.load_state_dict(torch.load(file_path))
        print(f"Веса успешно загрузились по пути {file_path}")

############################## ВСПОМОГАТЕЛЬНЫЙ ФУНКЦИИ ##############################

def computing_device() -> torch.device:
    """
    Функция, которая проверяет возможность переноса вычислений на GPU
    - Args:
      - отсутствуют
    - Return:
      - вычислительное устройство
    """
    print(f"Доступность CUDA: {torch.cuda.is_available()}")
    #определяем вычислительное устройство (CPU или GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"Обработка вычислений будет происходить на {device_name}")
    return device

def create_base_components(input_size: int, hidden_size: int, output_size: int, device: torch.device) -> Tuple["LSTM_Network", torch.nn.CrossEntropyLoss, torch.optim.SGD]:
    """"
    Функция, которая создаёт сеть, функцию потерь и оптимизатор
    - Args:
      - input_size - размерность входного слоя
      - hidden_size - размерность скрытого слоя
      - output_size - размерность выходного слоя
      - device - устройство-обработчик вычислений
    - Return:
      - сеть, функция потерь, оптимизатор
    """
    #создание нейронной сети
    network = LSTM_Network(input_size, hidden_size, output_size, device)
    #пробуем распределять вычисления на одном устройстве
    network = nn.DataParallel(network)
    #определение кросс-энтропийной функции потерь
    criterion = nn.CrossEntropyLoss()
    #определение оптимизатора (с обработкой на GPU)
    optimizer = optim.SGD(network.parameters(), lr=0.05, momentum=0.9)
    return network, criterion, optimizer

############################## ОСНОВНЫЕ ПАРАМЕТРЫ ДЛЯ НАСТРОЙКИ РАБОТЫ СЕТИ ##############################

#параметры слоёв сети
input_size = 2100 #входной слой
hidden_size = 8000 #скрытый слой
output_size = 2100 #выходной слой

#параметр обучения сети
epochs = 50 #количество эпох обучения
number_save = 5 #каждую эту эпоху будет происходить сохранение весов
batch_size = 96 #размер батча
number_of_workers = 0 #количество потоков обработки данных

#параметры генерации текста
number_of_results = 3 #количество результатов работы сети
input_text = "Придет к учителю, сядет и молчит и как будто что-то высматривает." #входной текст
max_length = int(len(input_text.split()) + 500) #максимальная длина результата
temperature = 7 #коэффициент разнообразия генерации

#параметры обучающих файлов и файлов весов
data_files_paths = ["APCh_data.txt",
                    "Anton_Pavlovich_Chekhov_data.txt"] #файлы с текстами авторов
weights_files_paths = ["APCh_weights_epoch_50_v10.pth"] #файлы с весами для авторов

############################## АЛГОРИТМ РАБОТЫ СЕТИ ##############################

#получение устройства-обработчика вычислений (либо просто CPU, либо CPU + GPU)
device = computing_device()
#создание сети, функции потерь и оптимизатора
network, loss_function, optimizer = create_base_components(input_size, hidden_size, output_size, device)
#пробуем перенести вычисления на GPU для ускорения (обращаемся к module, чтобы не конфликтовало с DataParallel)
new_network, new_loss_function, new_optimizer = network.module.trying_to_use_gpu(loss_function, optimizer)
#тест сети до обучения
#results_before = new_network.network_testing(number_of_results, input_text, "", device, max_length, temperature)
#загрузка весов
#new_network.load_network_weights(weights_files_paths[3])
#подготовка данных и тренировка
#new_network.data_preparation_and_training(data_files_paths[0], batch_size, number_of_workers, new_loss_function, new_optimizer, epochs, number_save, device)
#тест после обучения
#results_after = new_network.network_testing(number_of_results, input_text, "", device, max_length, temperature)
#печать результатов
#new_network.print_results(results_before, results_after)

############################## НЕБОЛЬШИЕ ЗАМЕТКИ ##############################

#Неплохие результаты для APCh_weights_epoch_50_v10.pth с параметрами: 2100, 8000, 2100, max_length + 500, temperature - 7

############################## СОЗДАНИЕ ОКНА ДЛЯ ВЗАИМОДЕЙСТВИЯ (ОСНОВНЫЕ ФУНКЦИИ) ##############################

def create_window() -> tk.Tk:
    """
    Функция, которая создаёт основное окно
    - Args:
      - отсутствуют
    - Return:
      - созданное окно
    """
    #создаём окно
    main_window = tk.Tk()
    #задаём окну название
    main_window.title("Авторская нейросеть")
    #задаём параметры окна
    main_window_width = 800
    main_window_height = 400
    center_window(main_window, main_window_width, main_window_height)
    #запрещаем пользователю растягивать окно
    main_window.resizable(False, False)
    return main_window

def center_window(window: tk.Tk, width: int, height: int) -> None:
    """
    Функция, которая центрирует окно
    - Args:
      - window - само окно
      - width - ширина окна
      - height - высота окна
    - Return:
      - отсутствует
    """
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    window.geometry(f"{width}x{height}+{x}+{y}")

def insert_result(result: str, output_textarea: tk.Text) -> None:
    """
    Функция, которая вставляет результат генерации в поле вывода
    - Args:
      - result - результат генерации
      - output_textarea - зона вывода результата
    - Return:
      - отсутствует
    """
    output_textarea.insert("1.0", result)

def input_handle_button(main_window: tk.Tk, combobox_authors: ttk.Combobox, input_textarea: tk.Text, label_info: tk.Label, output_textarea: tk.Text, label_progress: tk.Label) -> None:
    """
    Функция, которая получает данные из комбобокса и зоны ввода при нажатии на кнопку
    - Args:
      - main_window - основное окно, которое содержит все элементы
      - combobox_authors - комбобокс, который содержит авторов
      - input_textarea - зона ввода текста
      - label_info - метка-подсказка
      - output_textarea - зона вывода результата
      - label_progress - метка-прогресс генерации
    - Return:
      - отсутствует
    """
    selected_author = combobox_authors.get()
    input_text = input_textarea.get("1.0", "end-1c")
    if selected_author == "" or (not input_text or all(char in string.whitespace + string.punctuation for char in input_text)):
        label_info.config(text="Введены некорректные данные для генерации!")
    else:
        label_info.config(text="Приступаем к генерации!")
        if selected_author == "Антон Павлович Чехов":
            new_network.load_network_weights(weights_files_paths[0])
            max_length = int(len(input_text.split()) + 500)
            #запуск функции генерации в параллельном потоке, чтобы графический интерфейс не зависал
            response_thread = threading.Thread(target=new_network.network_testing, args=(1, input_text, selected_author, device, max_length, temperature, insert_result, output_textarea, label_progress))
            response_thread.start()

def create_base_components_window(main_window: tk.Tk) -> None:
    """
    Функция, которая создаёт все основные компоненты для работы с сетью
    - Args:
      - main_window - основное окно
    - Return:
      - отсутствует
    """
    #создаём комбобокс (запрещаем изменять с клавиатуры)
    combobox_authors = ttk.Combobox(main_window, state="readonly", width=25)
    #задаём содержимое
    combobox_authors["values"] = ("Антон Павлович Чехов",)
    #задаём местоположение
    combobox_authors.place(x=25, y=50)

    #создаём зону ввода текста
    input_textarea = tk.Text(main_window, wrap=tk.WORD, width=40, height=10)
    #задаём местоположение зоне ввода
    input_textarea.place(x=25, y=100)

    #создаём вертикальный скроллбар для зоны ввода текста
    input_vertical_scrollbar = tk.Scrollbar(main_window, orient="vertical", command=input_textarea.yview)
    #связываем его с зоной ввода текста
    input_textarea.config(yscrollcommand=input_vertical_scrollbar.set)
    #задаём ему местоположение
    input_vertical_scrollbar.place(x=345, y=100, height=165)

    #создаём метку-подсказку
    label_info = tk.Label(main_window, text="")
    label_info.place(x=25, y=330)

    #создаёт зону вывода текста (и запрещаем редактировать)
    output_textarea = tk.Text(main_window, wrap=tk.WORD, width=40, height=10)
    #задаём ей местоположение
    output_textarea.place(x=425, y=100)

    #создаём вертикальный скроллбар для зоны вывода текста
    output_vertical_scrollbar = tk.Scrollbar(main_window, orient="vertical", command=output_textarea.yview)
    #связываем его с зоной вывода текста
    output_textarea.config(yscrollcommand=output_vertical_scrollbar.set)
    #задаём ему местоположение
    output_vertical_scrollbar.place(x=745, y=100, height=165)

    #создаём кнопку, которая получает выбранного автора и введённый текст, а также обрабатывает данные и начинает генерацию
    generate_button = tk.Button(main_window, text="Сгенерировать", command=lambda: input_handle_button(main_window, combobox_authors, input_textarea, label_info, output_textarea, label_progress))
    generate_button.place(x=25, y=290)

    #создаём метку-прогресс генерации
    label_progress = tk.Label(main_window, text="")
    label_progress.place(x=425, y=50)

############################## АЛГОРИТМ РАБОТЫ ОКНА ##############################

#основное окно
main_window = create_window()
#создаём основные компоненты окна
create_base_components_window(main_window)
#запускаем основной цикл отображения
main_window.mainloop()
