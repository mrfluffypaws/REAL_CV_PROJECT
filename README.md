# <center> REAL_CV_PROJECT. Creating a virtual coach

## Оглавление  
[1. Описание проекта](https://github.com/mrfluffypaws/REAL_CV_PROJECT/blob/main/README.md#Описание-проекта)  
[2. Задачи](https://github.com/mrfluffypaws/REAL_CV_PROJECT/blob/main/README.md#Задачи)  
[3. Краткая информация о данных](https://github.com/mrfluffypaws/REAL_CV_PROJECT/blob/main/README.md#Краткая-информация-о-данных)  
[4. Этапы работы над проектом](https://github.com/mrfluffypaws/REAL_CV_PROJECT/blob/main/README.md#Этапы-работы-над-проектом)  
[5. Результат](https://github.com/mrfluffypaws/REAL_CV_PROJECT/blob/main/README.md#Результаты)    
[6. Выводы](https://github.com/mrfluffypaws/REAL_CV_PROJECT/blob/main/README.md#Выводы) 

### Описание проекта    
Создание виртуального коуча, способного анализировать действия человека на видео с помощью распознавания этих действия по ключевым точкам, обнаруженным на теле субъекта. 

:arrow_up:[к оглавлению](https://github.com/mrfluffypaws/REAL_CV_PROJECT/blob/main/README.md#Оглавление)


### Задачи    
Построение каркаса позы через ключевые точки;
Оценка сходства поз по фотографии;
Валидация оценки позы на видео;
Разработка своего виртуального коуча.


**Что практикуем**  
* Работу с изображениями и видео с помощью библиотек PyTorch и OpenCV;     
* Выравнивание и сопоставление изображений, в частности метод аффинного отображения;
* Методы косинусного сходства и взвешенного совпадения для оценки степени похожести между объектами.

:arrow_up:[к оглавлению](https://github.com/mrfluffypaws/REAL_CV_PROJECT/blob/main/README.md#Оглавление)


### Краткая информация о данных
В качестве исходных данных были взяты два изображения бегущих девушек, видеофайл, содержаний обучение танцам, а также видеофайл, содержащий силовую тренировку.   

Изображения (https://github.com/mrfluffypaws/REAL_CV_PROJECT/tree/main/src/myproject_models/images)
Видеофайлы (https://github.com/mrfluffypaws/REAL_CV_PROJECT/tree/main/src/myproject_models/video) 

  
:arrow_up:[к оглавлению](https://github.com/mrfluffypaws/REAL_CV_PROJECT/blob/main/README.md#Оглавление)


### Этапы работы над проектом  
1. Загрузка модели;
    1. В качестве модели была использована предобученная модель keypointrcnn_resnet50_fpn из библиотеки torchvision, способная обнаруживать объекты и предсказывает координаты их ключевых точек.
2. Визуализация ключевых точек и построение скелета;
3. Выравнивание и сопоставление изображений с отображением ключевых метрик (косинусное сходство, взвешенное совпадение);
4. Работа с видео на примере ролика обучающего танцам;
5. Итоговое сравнение на примере ролика силовой тренировки.


:arrow_up:[к оглавлению](https://github.com/mrfluffypaws/REAL_CV_PROJECT/blob/main/README.md#Оглавление)


### Результаты:  
В процессе решения поставленных задач отработаны методы работы с изображениями, методы выявления ключевых точек объектов, построение скелетов объектов, сравнение поз объектов с использованием косинусного сходства и взвешенного совпадения, а также работа с видео файлами и проведение аналогичных манипуляций с ними.

:arrow_up:[к оглавлению](https://github.com/mrfluffypaws/REAL_CV_PROJECT/blob/main/README.md#Оглавление)


### Выводы:  
В результате был создан виртуальный коуч (финальное видео), способный сопоставлять движения объекта (человека) с собственными движениями (эталонное видео).      

Результат работы - видеофайл Final_result.mp4 (https://github.com/mrfluffypaws/REAL_CV_PROJECT/tree/main/src/myproject_models/video)

:arrow_up:[к оглавлению](https://github.com/mrfluffypaws/REAL_CV_PROJECT/blob/main/README.md#Оглавление)


Если информация по этому проекту покажется вам интересной или полезной, то я буду очень вам благодарен, если отметите репозиторий и профиль ⭐️⭐️⭐️-дами

