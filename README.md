# shortest_path
Skoltech hackathon "Shortest path" solution

Решение состояло из нескольких шагов:

1. Отрисовка в виде прямоугольных линий.

![My_img](https://github.com/elejke/shortest_path/blob/master/pics/squared_all.png?raw=true)

2. Выделение наибольших по вложению последовательностей. Решение с помощью проектирования координат на одну ось и использования алгоритма поиска максимальной возрастающей подпоследовательности. Результат - наборы таких соединений (на картинке отмечены одним цветом):

![My_img](https://github.com/elejke/shortest_path/blob/master/pics/lis_all.png?raw=true)

3. Предложение использовать по 2 набора таких непересекающихся соединений для укладки на один слой в таком виде:

![My_img](https://github.com/elejke/shortest_path/blob/master/pics/embedding_nosq_test.png?raw=true)

4. Оптимизация длины путём срезания углов.

![My_img](https://github.com/elejke/shortest_path/blob/master/pics/embedding_with_turns.png?raw=true)

5. Дополнительная оптимизация - проведение некоторых соединений из внешнего (нижнего) круга через верх от внутренних.

![My_img](https://github.com/elejke/shortest_path/blob/master/pics/pre-final.png?raw=true)

6. Окончательная укладка - проведение мелких оптимизаций по срезанию длины линий.

![My_img](https://github.com/elejke/shortest_path/blob/master/pics/final.png?raw=true)


![My_img](https://github.com/elejke/shortest_path/blob/master/pics/squared_all.png?raw=true)

2. Выделение наибольших по вложению последовательностей. Решение с помощью проектирования координат на одну ось и использования алгоритма поиска максимальной возрастающей подпоследовательности. Результат - наборы таких соединений (на картинке отмечены одним цветом):

![My_img](https://github.com/elejke/shortest_path/blob/master/pics/lis_all.png?raw=true)

3. Предложение использовать по 2 набора таких непересекающихся соединений для укладки на один слой в таком виде:

![My_img](https://github.com/elejke/shortest_path/blob/master/pics/embedding_nosq_test.png?raw=true)

4. Оптимизация длины путём срезания углов.

![My_img](https://github.com/elejke/shortest_path/blob/master/pics/embedding_with_turns.png?raw=true)

5. Дополнительная оптимизация - проведение некоторых соединений из внешнего (нижнего) круга через верх от внутренних.

![My_img](https://github.com/elejke/shortest_path/blob/master/pics/pre-final.png?raw=true)

6. Окончательная укладка - проведение мелких оптимизаций по срезанию длины линий.

![My_img](https://github.com/elejke/shortest_path/blob/master/pics/final.png?raw=true)
