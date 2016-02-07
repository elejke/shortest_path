# shortest_path
Skoltech hackathon "Shortest path" solution

Решение состояло из нескольких шагов:

1. Отрисовка в виде прямоугольных линий

![My_img](https://raw.githubusercontent.com/elejke/shortest_path/master/pics/squared_all.png?token=AGcTWvGTn2Sy-BZK_DD6nGPXzodQRGU4ks5WwRJAwA%3D%3D)

2.1 Предложение выделять наибольшие по вложению последовательности

2.2 Решение с помощью проектирования координат на одну ось и использования алгоритма поиска максимальной возрастающей подпоследовательности. Результат - наборы таких соединений (на картинке отмечены одним цветом):

![My_img](https://raw.githubusercontent.com/elejke/shortest_path/master/pics/lis_all.png?token=AGcTWngi7ltELIjl5zpGu-GdUkYH0nC4ks5WwRNOwA%3D%3D)

3. Предложение использовать по 2 набора таких непересекающихся соединений для укладки на один слой в таком виде:

pic

4. Оптимизация длины путём срезания углов.

pic

5. Дополнительная оптимизация - проведение некоторых соединений из внешнего (нижнего) круга через верх от внутренних.

pic

6. Окончательная укладка
