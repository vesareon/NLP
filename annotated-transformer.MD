# Аннотированный Transformer

[Attention is All You Need](https://arxiv.org/abs/1706.03762)

- *v2022: Austin Huang, Suraj Subramanian, Jonathan Sum, Khalid Almubarak, Stella Biderman.*
- *[Оригинал](https://nlp.seas.harvard.edu/2018/04/03/attention.html): [Sasha Rush](http://rush-nlp.com/).*

Transformer был на устах многих людей в течение последнего ~~года~~ пяти лет. Этот пост представляет аннотированную версию статьи в форме построчной реализации. Он переупорядочивает и удаляет некоторые разделы из оригинальной статьи и добавляет комментарии по всему тексту. Этот документ сам по себе является рабочим блокнотом и должен быть полностью работоспособной реализацией. Код доступен [здесь](https://github.com/harvardnlp/annotated-transformer/).

### Содержание

- Преамбула
- Предыстория
- Часть 1: Архитектура модели
- Архитектура модели
  - Стек кодера и декодера
- Позиционные полносвязные сети
- Векторные представления и Softmax
- Позиционное кодирование
- Полная модель
- Инференс
- Часть 2: Обучение модели
- Обучение
  - Пакеты и маскирование
  - Цикл обучения
  - Обучающие данные и пакетирование
  - Оборудование и расписание
  - Оптимизатор
  - Регуляризация
- Первый пример
  - Синтетические данные
  - Вычисление ошибки
  - Жадный декодинг
- Часть 3: Пример из реального мира
  - Загрузка данных
  - Итераторы
  - Обучение системы
- Дополнительные компоненты: BPE, поиск, усреднение
- Результаты
  - Визуализация внимания
  - Самовнимание кодера
  - Самовнимание декодера
  - Внимание декодера к источнику
- Заключение

# Преамбула

Пропустить

```python
# !pip install -r requirements.txt

# # Раскомментировать для colab
# #
# !pip install -q torchdata==0.3.0 torchtext==0.12 spacy==3.2 altair GPUtil
# !python -m spacy download de_core_news_sm
# !python -m spacy download en_core_web_sm

import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# Установите False, чтобы пропустить выполнение блокнота (например, для отладки)
warnings.filterwarnings("ignore")
RUN_EXAMPLES = True

# Некоторые вспомогательные функции, используемые в блокноте

def is_interactive_notebook():
    return __name__ == "__main__"

def show_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)

def execute_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        fn(*args)

class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None

class DummyScheduler:
    def step(self):
        None
```

> Мои комментарии выделены цитированием. Основной текст взят из самой статьи.

# Предыстория

Цель уменьшения последовательных вычислений также лежит в основе Extended Neural GPU, ByteNet и ConvS2S, все из которых используют сверточные нейронные сети в качестве базового строительного блока, вычисляя скрытые представления параллельно для всех входных и выходных позиций. В этих моделях количество операций, необходимых для связи сигналов от двух произвольных входных или выходных позиций, растет с расстоянием между позициями — линейно для ConvS2S и логарифмически для ByteNet. Это затрудняет изучение зависимостей между удаленными позициями. В Transformer это количество операций сведено к константе, хотя и ценой снижения эффективного разрешения из-за усреднения позиций, взвешенных по вниманию, — эффект, который мы компенсируем многоголовым вниманием.

Самовнимание, иногда называемое внутри-вниманием, — это механизм внимания, связывающий различные позиции одной последовательности для вычисления представления последовательности. Самовнимание успешно применялось в различных задачах, включая понимание прочитанного, абстрактивное реферирование, текстовое следование и изучение независимых от задачи представлений предложений. Сквозные сети памяти основаны на рекуррентном механизме внимания вместо рекуррентности, выровненной по последовательности, и показали хорошие результаты на задачах вопросно-ответных систем на простом языке и моделирования языка.

Насколько нам известно, однако, Transformer — это первая трансукционная модель, полностью полагающаяся на самовнимание для вычисления представлений своего входа и выхода без использования RNN или сверток, выровненных по последовательности.

# Часть 1: Архитектура модели

# Архитектура модели

Большинство конкурентоспособных нейронных моделей последовательной трансукции имеют структуру кодер-декодер [(ссылка)](https://arxiv.org/abs/1409.0473). Здесь кодер отображает входную последовательность символьных представлений (x_1, ..., x_n) в последовательность непрерывных представлений $\mathbf{z}$ = (z_1, ..., z_n). Затем декодер генерирует выходную последовательность (y_1,...,y_m) символов по одному элементу за раз. На каждом шаге модель является авторегрессивной [(ссылка)](https://arxiv.org/abs/1308.0850), потребляя ранее сгенерированные символы в качестве дополнительного входа при генерации следующего.

```python
class EncoderDecoder(nn.Module):
    """
    Стандартная архитектура кодер-декодер. Основа для этой и многих
    других моделей.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Принимает и обрабатывает замаскированные src и tgt последовательности."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    "Определяет стандартный шаг генерации linear + softmax."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)
```

Transformer следует этой общей архитектуре, используя стек самовнимания и поточечные полносвязные слои как для кодера, так и для декодера, показанные соответственно в левой и правой половинах Рисунка 1.
<p align="center">
  <img src="https://media.geeksforgeeks.org/wp-content/uploads/20251004125140134507/transformers.webp" width="400" alt="Описание картинки">
</p>

## Стек кодера и декодера

### Кодер

Кодер состоит из стека N=6 идентичных слоев.

```python
def clones(module, N):
    "Производит N идентичных слоев."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Основной кодер — это стек из N слоев"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Пропускает входные данные (и маску) через каждый слой по очереди."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```

Мы используем остаточное соединение [(ссылка)](https://arxiv.org/abs/1512.03385) вокруг каждого из двух подслоев, за которым следует нормализация слоя [(ссылка)](https://arxiv.org/abs/1607.06450).

```python
class LayerNorm(nn.Module):
    "Создает модуль layernorm (см. ссылку для деталей)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

То есть, выход каждого подслоя равен $\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$, где $\mathrm{Sublayer}(x)$ — функция, реализованная самим подслоем. Мы применяем dropout [(ссылка)](http://jmlr.org/papers/v15/srivastava14a.html) к выходу каждого подслоя перед его добавлением ко входу подслоя и нормализацией.

Чтобы облегчить эти остаточные соединения, все подслои в модели, а также слои векторных представлений, производят выходы размерности $d_{\text{model}}$=512.

```python
class SublayerConnection(nn.Module):
    """
    Остаточное соединение, за которым следует нормализация слоя.
    Для простоты кода нормализация выполняется первой, а не последней.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Применяет остаточное соединение к любому подслою того же размера."
        return x + self.dropout(sublayer(self.norm(x)))
```

Каждый слой имеет два подслоя. Первый — это механизм многоголового самовнимания, а второй — простая позиционная полносвязная сеть прямого распространения.

```python
class EncoderLayer(nn.Module):
    "Кодер состоит из самовнимания и прямого распространения (определено ниже)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Следуйте Рисунку 1 (слева) для соединений."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
```

### Декодер

Декодер также состоит из стека N=6 идентичных слоев.

```python
class Decoder(nn.Module):
    "Общий N-слойный декодер с маскированием."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
```

В дополнение к двум подслоям в каждом слое кодера, декодер добавляет третий подслой, который выполняет многоголовое внимание к выходу стека кодера. Как и в кодере, мы используем остаточные соединения вокруг каждого из подслоев, за которыми следует нормализация слоя.

```python
class DecoderLayer(nn.Module):
    "Декодер состоит из self-attn, src-attn и прямого распространения (определено ниже)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Следуйте Рисунку 1 (справа) для соединений."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
```

Мы также модифицируем подслой самовнимания в стеке декодера, чтобы предотвратить внимание позиций к последующим позициям. Это маскирование, в сочетании с тем фактом, что выходные векторные представления сдвинуты на одну позицию, гарантирует, что предсказания для позиции i могут зависеть только от известных выходов на позициях, меньших i.

```python
def subsequent_mask(size):
    "Маскирует последующие позиции."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0
```

> Маска внимания ниже показывает, на какие позиции (столбцы) разрешено смотреть каждому слову цели (строка). Словам запрещено обращать внимание на будущие слова во время обучения.

```python
def example_mask():
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Subsequent Mask": subsequent_mask(20)[0][x, y].flatten(),
                    "Window": y,
                    "Masking": x,
                }
            )
            for y in range(20)
            for x in range(20)
        ]
    )

    return (
        alt.Chart(LS_data)
        .mark_rect()
        .properties(height=250, width=250)
        .encode(
            alt.X("Window:O"),
            alt.Y("Masking:O"),
            alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),
        )
        .interactive()
    )

show_example(example_mask)
```
<p align="center">
  <img src="https://nlp.seas.harvard.edu/images/the-annotated-transformer_31_0.png" width="400" alt="Описание картинки">
</p>


### Внимание

Функцию внимания можно описать как отображение запроса и набора пар ключ-значение на выход, где запрос, ключи, значения и выход — все являются векторами. Выход вычисляется как взвешенная сумма значений, где вес, присвоенный каждому значению, вычисляется функцией совместимости запроса с соответствующим ключом.

Мы называем наше конкретное внимание «масштабированным скалярным произведением внимания» (Scaled Dot-Product Attention). Вход состоит из запросов и ключей размерности $d_k$ и значений размерности $d_v$. Мы вычисляем скалярные произведения запроса со всеми ключами, делим каждое на $\sqrt{d_k}$ и применяем функцию softmax для получения весов значений.

<p align="center">
  <img src="https://pub.mdpi-res.com/sensors/sensors-22-04639/article_deploy/html/images/sensors-22-04639-g001-550.jpg?1655711832" width="200" alt="Описание картинки">
</p>


На практике мы вычисляем функцию внимания для набора запросов одновременно, упакованных вместе в матрицу Q. Ключи и значения также упаковываются вместе в матрицы K и V. Мы вычисляем матрицу выходов как:


$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$


```python
def attention(query, key, value, mask=None, dropout=None):
    "Вычисляет 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
```

Две наиболее часто используемые функции внимания — это аддитивное внимание [(ссылка)](https://arxiv.org/abs/1409.0473) и скалярное произведение (мультипликативное) внимание. Внимание через скалярное произведение идентично нашему алгоритму, за исключением масштабирующего коэффициента $\frac{1}{\sqrt{d_k}}$. Аддитивное внимание вычисляет функцию совместимости с помощью сети прямого распространения с одним скрытым слоем. Хотя эти два механизма схожи по теоретической сложности, внимание через скалярное произведение на практике гораздо быстрее и эффективнее по памяти, поскольку может быть реализовано с использованием высокооптимизированного кода умножения матриц.

В то время как для малых значений $d_k$ оба механизма работают одинаково, аддитивное внимание превосходит внимание через скалярное произведение без масштабирования для больших значений $d_k$ [(ссылка)](https://arxiv.org/abs/1703.03906). Мы подозреваем, что для больших значений $d_k$ скалярные произведения становятся большими по величине, что приводит softmax в области с чрезвычайно малыми градиентами (Чтобы проиллюстрировать, почему скалярные произведения становятся большими, предположим, что компоненты q и k являются независимыми случайными величинами со средним 0 и дисперсией 1. Тогда их скалярное произведение, q $\cdot$ k = $\sum_{i=1}^{d_k}$ q_ik_i, имеет среднее 0 и дисперсию $d_k$.). Чтобы противодействовать этому эффекту, мы масштабируем скалярные произведения на $\frac{1}{\sqrt{d_k}}$.

<p align="center">
  <img src="https://i.pinimg.com/736x/f4/61/65/f4616545a4beab3e8230766e2127d271.jpg" width="700" alt="Описание картинки">
</p>


Многоголовое внимание позволяет модели совместно обращать внимание на информацию из разных подпространств представлений на разных позициях. С одной головой внимания усреднение препятствует этому.

$$
\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head_1}, ..., \mathrm{head_h})W^O\ \text{где}~\mathrm{head_i} = \mathrm{Attention}(QW^Q_i, KW^K_i, VW^V_i)
$$


Где проекции — это матрицы параметров $W^Q_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W^K_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W^V_i \in \mathbb{R}^{d_{\text{model}} \times d_v}$ и $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$.

В этой работе мы используем h=8 параллельных слоев внимания, или голов. Для каждой из них мы используем $d_k=d_v=d_{\text{model}}/h=64$. Благодаря уменьшенной размерности каждой головы, общая вычислительная стоимость аналогична стоимости однослойного внимания с полной размерностью.

```python
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Принимает размер модели и количество голов."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # Мы предполагаем, что d_v всегда равно d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Реализует Рисунок 2"
        if mask is not None:
            # Одна и та же маска применяется ко всем h головам.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Выполняем все линейные проекции в пакете из d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Применяем внимание ко всем спроецированным векторам в пакете.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" с помощью view и применяем финальный линейный слой.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)
```

### Применение внимания в нашей модели

Transformer использует многоголовое внимание тремя различными способами: 1) В слоях «внимание кодер-декодер» запросы поступают с предыдущего слоя декодера, а ключи и значения памяти — с выхода кодера. Это позволяет каждой позиции в декодере обращать внимание на все позиции во входной последовательности. Это имитирует типичные механизмы внимания кодер-декодер в моделях последовательность-в-последовательность, таких как [(ссылка)](https://arxiv.org/abs/1609.08144).

2. Кодер содержит слои самовнимания. В слое самовнимания все ключи, значения и запросы поступают из одного и того же места, в данном случае с выхода предыдущего слоя кодера. Каждая позиция в кодере может обращать внимание на все позиции предыдущего слоя кодера.

3. Аналогично, слои самовнимания в декодере позволяют каждой позиции в декодере обращать внимание на все позиции в декодере вплоть до и включая эту позицию. Нам необходимо предотвратить поток информации слева направо в декодере, чтобы сохранить авторегрессионное свойство. Мы реализуем это внутри масштабированного скалярного произведения внимания, маскируя (устанавливая в -$\infty$) все значения на входе softmax, которые соответствуют недопустимым соединениям.

## Позиционные полносвязные сети

В дополнение к подслоям внимания, каждый из слоев в нашем кодере и декодере содержит полносвязную сеть прямого распространения, которая применяется к каждой позиции отдельно и идентично. Она состоит из двух линейных преобразований с активацией ReLU между ними.


$$
\mathrm{FFN}(x)=\max(0, xW_1 + b_1) W_2 + b_2
$$


Хотя линейные преобразования одинаковы для разных позиций, они используют разные параметры от слоя к слою. Другой способ описания — это две свертки с размером ядра 1. Размерность входа и выхода равна $d_{\text{model}}$=512, а размерность внутреннего слоя равна $d_{ff}$=2048.

```python
class PositionwiseFeedForward(nn.Module):
    "Реализует уравнение FFN."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
```

## Векторные представления и Softmax

Аналогично другим моделям последовательной трансукции, мы используем изучаемые векторные представления для преобразования входных и выходных токенов в векторы размерности $d_{\text{model}}$. Мы также используем обычное изучаемое линейное преобразование и функцию softmax для преобразования выхода декодера в предсказанные вероятности следующего токена. В нашей модели мы используем общую матрицу весов между двумя слоями векторных представлений и линейным преобразованием перед softmax, аналогично [(ссылка)](https://arxiv.org/abs/1608.05859). В слоях векторных представлений мы умножаем эти веса на \sqrt{$d_{\text{model}}$}.

```python
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
```

## Позиционное кодирование

Поскольку наша модель не содержит рекуррентности и свертки, для того чтобы модель могла использовать порядок последовательности, мы должны внедрить некоторую информацию об относительном или абсолютном положении токенов в последовательности. С этой целью мы добавляем «позиционные кодировки» к входным векторным представлениям в нижней части стека кодера и декодера. Позиционные кодировки имеют ту же размерность $d_{\text{model}}$, что и векторные представления, так что они могут быть суммированы. Существует много вариантов позиционных кодировок, как изучаемых, так и фиксированных [(ссылка)](https://arxiv.org/pdf/1705.03122.pdf).

В этой работе мы используем синусные и косинусные функции разных частот:


$$
PE_{(pos,2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})
$$



$$
PE_{(pos,2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})
$$


где pos — позиция, а i — размерность. То есть, каждое измерение позиционной кодировки соответствует синусоиде. Длины волн образуют геометрическую прогрессию от $2\pi$ до 10000 $\cdot$ $2\pi$. Мы выбрали эту функцию, потому что предположили, что она позволит модели легко научиться обращать внимание по относительным позициям, поскольку для любого фиксированного сдвига k, PE_{pos+k} может быть представлена как линейная функция от PE_{pos}.

Кроме того, мы применяем dropout к суммам векторных представлений и позиционных кодировок как в стеке кодера, так и декодера. Для базовой модели мы используем коэффициент $P_{drop}$=0.1.

```python
class PositionalEncoding(nn.Module):
    "Реализует функцию PE."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Вычисляем позиционные кодировки один раз в логарифмическом пространстве.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
```

> Ниже позиционное кодирование добавляет синусоидальную волну на основе позиции. Частота и сдвиг волны различны для каждого измерения.

```python
def example_positional():
    pe = PositionalEncoding(20, 0)
    y = pe.forward(torch.zeros(1, 100, 20))

    data = pd.concat(
        [
            pd.DataFrame(
                {
                    "embedding": y[0, :, dim],
                    "dimension": dim,
                    "position": list(range(100)),
                }
            )
            for dim in [4, 5, 6, 7]
        ]
    )

    return (
        alt.Chart(data)
        .mark_line()
        .properties(width=800)
        .encode(x="position", y="embedding", color="dimension:N")
        .interactive()
    )

show_example(example_positional)
```

<p align="center">
  <img src="https://nlp.seas.harvard.edu/images/the-annotated-transformer_49_0.png" width="1000" alt="Описание картинки">
</p>

Мы также экспериментировали с использованием изучаемых позиционных векторных представлений [(ссылка)](https://arxiv.org/pdf/1705.03122.pdf) и обнаружили, что две версии дают почти идентичные результаты. Мы выбрали синусоидальную версию, потому что она может позволить модели экстраполировать на длины последовательностей, превышающие те, что встречались во время обучения.

## Полная модель

> Здесь мы определяем функцию от гиперпараметров к полной модели.

```python
def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    "Вспомогательная функция: создает модель из гиперпараметров."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # Это было важно из их кода.
    # Инициализируем параметры с помощью Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
```

## Инференс

> Здесь мы делаем прямой шаг для генерации предсказания модели. Мы пытаемся использовать наш transformer для запоминания входных данных. Как вы увидите, выход генерируется случайно из-за того, что модель еще не обучена. В следующем уроке мы построим функцию обучения и попытаемся обучить нашу модель запоминать числа от 1 до 10.

```python
def inference_test():
    test_model = make_model(11, 11, 2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Пример предсказания необученной модели:", ys)

def run_tests():
    for _ in range(10):
        inference_test()

show_example(run_tests)
```

```
Пример предсказания необученной модели: tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
Пример предсказания необученной модели: tensor([[0, 3, 4, 4, 4, 4, 4, 4, 4, 4]])
Пример предсказания необученной модели: tensor([[ 0, 10, 10, 10,  3,  2,  5,  7,  9,  6]])
Пример предсказания необученной модели: tensor([[ 0,  4,  3,  6, 10, 10,  2,  6,  2,  2]])
Пример предсказания необученной модели: tensor([[ 0,  9,  0,  1,  5, 10,  1,  5, 10,  6]])
Пример предсказания необученной модели: tensor([[ 0,  1,  5,  1, 10,  1, 10, 10, 10, 10]])
Пример предсказания необученной модели: tensor([[ 0,  1, 10,  9,  9,  9,  9,  9,  1,  5]])
Пример предсказания необученной модели: tensor([[ 0,  3,  1,  5, 10, 10, 10, 10, 10, 10]])
Пример предсказания необученной модели: tensor([[ 0,  3,  5, 10,  5, 10,  4,  2,  4,  2]])
Пример предсказания необученной модели: tensor([[0, 5, 6, 2, 5, 6, 2, 6, 2, 2]])
```

# Часть 2: Обучение модели

# Обучение

Этот раздел описывает режим обучения для наших моделей.

> Мы делаем короткую паузу, чтобы представить некоторые инструменты, необходимые для обучения стандартной модели кодер-декодер. Сначала мы определяем объект пакета, который содержит исходные и целевые предложения для обучения, а также создает маски.

## Пакеты и маскирование

```python
class Batch:
    """Объект для хранения пакета данных с маской во время обучения."""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Создает маску для скрытия заполнения и будущих слов."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask
```

> Далее мы создаем универсальную функцию обучения и подсчета ошибок для отслеживания потерь. Мы передаем универсальную функцию вычисления потерь, которая также обрабатывает обновления параметров.

## Цикл обучения

```python
class TrainState:
    """Отслеживает количество шагов, примеров и обработанных токенов"""

    step: int = 0  # Шаги в текущей эпохе
    accum_step: int = 0  # Количество шагов накопления градиента
    samples: int = 0  # общее количество использованных примеров
    tokens: int = 0  # общее количество обработанных токенов

def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
):
    """Обучает одну эпоху"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Шаг эпохи: %6d | Шаг накопления: %3d | Потери: %6.2f "
                    + "| Токенов/сек: %7.1f | Скорость обучения: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state
```

## Обучающие данные и пакетирование

Мы обучались на стандартном наборе данных WMT 2014 английский-немецкий, состоящем из примерно 4.5 миллионов пар предложений. Предложения были закодированы с использованием байт-парного кодирования (BPE), которое имеет общий словарь источника и цели из около 37000 токенов. Для английского-французского мы использовали значительно больший набор данных WMT 2014 английский-французский, состоящий из 36M предложений, и разделили токены на словарь из 32000 подслов.

Пары предложений были сгруппированы в пакеты по приблизительной длине последовательности. Каждый обучающий пакет содержал набор пар предложений, содержащих примерно 25000 исходных токенов и 25000 целевых токенов.

## Оборудование и расписание

Мы обучали наши модели на одной машине с 8 GPU NVIDIA P100. Для наших базовых моделей, использующих гиперпараметры, описанные в статье, каждый шаг обучения занимал около 0.4 секунды. Мы обучали базовые модели в общей сложности 100000 шагов или 12 часов. Для наших больших моделей время шага составляло 1.0 секунды. Большие модели обучались в течение 300000 шагов (3.5 дня).

## Оптимизатор

Мы использовали оптимизатор Adam [(ссылка)](https://arxiv.org/abs/1412.6980) с $\beta_1$=0.9, $\beta_2$=0.98 и $\epsilon=10^{-9}$. Мы изменяли скорость обучения в процессе тренировки в соответствии с формулой:


$$
lrate = d_{\text{model}}^{-0.5} \cdot \min({step\\_num}^{-0.5}, {step\\_num} \cdot {warmup\\_steps}^{-1.5})
$$


Это соответствует линейному увеличению скорости обучения для первых warmup\\_steps шагов обучения и последующему уменьшению пропорционально обратному квадратному корню номера шага. Мы использовали warmup\\_steps=4000.

> Примечание: Эта часть очень важна. Необходимо обучать с такой настройкой модели.

> Пример кривых этой модели для различных размеров модели и гиперпараметров оптимизации.

```python
def rate(step, model_size, factor, warmup):
    """
    Мы должны установить шаг по умолчанию в 1 для функции LambdaLR,
    чтобы избежать возведения нуля в отрицательную степень.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

def example_learning_schedule():
    opts = [
        [512, 1, 4000],  # пример 1
        [512, 1, 8000],  # пример 2
        [256, 1, 4000],  # пример 3
    ]

    dummy_model = torch.nn.Linear(1, 1)
    learning_rates = []

    # у нас есть 3 примера в списке opts.
    for idx, example in enumerate(opts):
        # запускаем 20000 эпох для каждого примера
        optimizer = torch.optim.Adam(
            dummy_model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9
        )
        lr_scheduler = LambdaLR(
            optimizer=optimizer, lr_lambda=lambda step: rate(step, *example)
        )
        tmp = []
        # делаем 20K фиктивных шагов обучения, сохраняем скорость обучения на каждом шаге
        for step in range(20000):
            tmp.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            lr_scheduler.step()
        learning_rates.append(tmp)

    learning_rates = torch.tensor(learning_rates)

    # Включаем altair для обработки более 5000 строк
    alt.data_transformers.disable_max_rows()

    opts_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Learning Rate": learning_rates[warmup_idx, :],
                    "model_size:warmup": ["512:4000", "512:8000", "256:4000"][
                        warmup_idx
                    ],
                    "step": range(20000),
                }
            )
            for warmup_idx in [0, 1, 2]
        ]
    )

    return (
        alt.Chart(opts_data)
        .mark_line()
        .properties(width=600)
        .encode(x="step", y="Learning Rate", color="model_size:warmup:N")
        .interactive()
    )

example_learning_schedule()
```

<p align="center">
  <img src="https://nlp.seas.harvard.edu/images/the-annotated-transformer_69_0.png" width="600" alt="Описание картинки">
</p>

## Регуляризация

### Сглаживание меток

Во время обучения мы использовали сглаживание меток со значением $\epsilon_{ls}$=0.1 [(ссылка)](https://arxiv.org/abs/1512.00567). Это вредит перплексии, так как модель учится быть более неуверенной, но улучшает точность и BLEU-оценку.

> Мы реализуем сглаживание меток с использованием KL-дивергенции. Вместо использования one-hot целевого распределения, мы создаем распределение, которое имеет `уверенность` в правильном слове, а остальная масса `сглаживания` распределяется по всему словарю.

```python
class LabelSmoothing(nn.Module):
    "Реализует сглаживание меток."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())
```

> Здесь мы видим пример того, как масса распределяется по словам на основе уверенности.

```python
# Пример сглаживания меток.

def example_label_smoothing():
    crit = LabelSmoothing(5, 0, 0.4)
    predict = torch.FloatTensor(
        [
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
        ]
    )
    crit(x=predict.log(), target=torch.LongTensor([2, 1, 0, 3, 3]))
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "target distribution": crit.true_dist[x, y].flatten(),
                    "columns": y,
                    "rows": x,
                }
            )
            for y in range(5)
            for x in range(5)
        ]
    )

    return (
        alt.Chart(LS_data)
        .mark_rect(color="Blue", opacity=1)
        .properties(height=200, width=200)
        .encode(
            alt.X("columns:O", title=None),
            alt.Y("rows:O", title=None),
            alt.Color(
                "target distribution:Q", scale=alt.Scale(scheme="viridis")
            ),
        )
        .interactive()
    )

show_example(example_label_smoothing)
```

> Сглаживание меток фактически начинает штрафовать модель, если она становится очень уверенной в данном выборе.

```python
def loss(x, crit):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]])
    return crit(predict.log(), torch.LongTensor([1])).data

def penalization_visualization():
    crit = LabelSmoothing(5, 0, 0.1)
    loss_data = pd.DataFrame(
        {
            "Loss": [loss(x, crit) for x in range(1, 100)],
            "Steps": list(range(99)),
        }
    ).astype("float")

    return (
        alt.Chart(loss_data)
        .mark_line()
        .properties(width=350)
        .encode(
            x="Steps",
            y="Loss",
        )
        .interactive()
    )

show_example(penalization_visualization)
```

# Первый пример

> Мы можем начать с простой задачи копирования. Имея случайный набор входных символов из небольшого словаря, цель состоит в том, чтобы сгенерировать те же самые символы.

## Синтетические данные

```python
def data_gen(V, batch_size, nbatches):
    "Генерирует случайные данные для задачи копирования src-tgt."
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)
```

## Вычисление ошибки

```python
class SimpleLossCompute:
    "Простая функция вычисления потерь и обучения."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss
```

## Жадный декодинг

> Этот код предсказывает перевод с использованием жадного декодинга для простоты.

```python
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys

# Обучаем простую задачу копирования.

def example_simple_model():
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
        ),
    )

    batch_size = 80
    for epoch in range(20):
        model.train()
        run_epoch(
            data_gen(V, batch_size, 20),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
        )
        model.eval()
        run_epoch(
            data_gen(V, batch_size, 5),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )[0]

    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len)
    print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))

# execute_example(example_simple_model)
```

# Часть 3: Пример из реального мира

> Теперь рассмотрим пример из реального мира, используя задачу перевода Multi30k немецкий-английский. Эта задача намного меньше, чем задача WMT, рассмотренная в статье, но она иллюстрирует всю систему. Мы также показываем, как использовать многопроцессорную обработку GPU, чтобы сделать ее действительно быстрой.

## Загрузка данных

> Мы загрузим набор данных, используя torchtext и spacy для токенизации.

```python
# Загружаем модели токенизаторов spacy, скачиваем их, если они еще не были скачаны

def load_tokenizers():

    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en

def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]

def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])

def build_vocabulary(spacy_de, spacy_en):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    print("Построение немецкого словаря ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_de, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Построение английского словаря ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt

def load_vocab(spacy_de, spacy_en):
    if not exists("vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    print("Готово.\nРазмеры словарей:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt

if is_interactive_notebook():
    # глобальные переменные, используемые позже в скрипте
    spacy_de, spacy_en = show_example(load_tokenizers)
    vocab_src, vocab_tgt = show_example(load_vocab, args=[spacy_de, spacy_en])
```

```
Готово.
Размеры словарей:
59981
36745
```

> Пакетирование очень важно для скорости. Мы хотим иметь очень равномерно разделенные пакеты с минимальным дополнением. Для этого нам нужно немного обойти стандартное пакетирование torchtext. Этот код модифицирует их стандартное пакетирование, чтобы гарантировать, что мы просматриваем достаточно предложений для поиска плотных пакетов.

## Итераторы

```python
def collate_batch(
    batch,
    src_pipeline,
    tgt_pipeline,
    src_vocab,
    tgt_vocab,
    device,
    max_padding=128,
    pad_id=2,
):
    bs_id = torch.tensor([0], device=device)  # токен <s>
    eos_id = torch.tensor([1], device=device)  # токен </s>
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(src_pipeline(_src)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tgt_pipeline(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            # предупреждение - перезаписывает значения для отрицательных значений padding - len
            pad(
                processed_src,
                (
                    0,
                    max_padding - len(processed_src),
                ),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)

def create_dataloaders(
    device,
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    batch_size=12000,
    max_padding=128,
    is_distributed=True,
):
    # def create_dataloaders(batch_size=12000):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    train_iter, valid_iter, test_iter = datasets.Multi30k(
        language_pair=("de", "en")
    )

    train_iter_map = to_map_style_dataset(
        train_iter
    )  # DistributedSampler нуждается в len(dataset)
    train_sampler = (
        DistributedSampler(train_iter_map) if is_distributed else None
    )
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = (
        DistributedSampler(valid_iter_map) if is_distributed else None
    )

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader
```

## Обучение системы

```python
def train_worker(
    gpu,
    ngpus_per_node,
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    config,
    is_distributed=False,
):
    print(f"Процесс обучения использует GPU: {gpu} для обучения", flush=True)
    torch.cuda.set_device(gpu)

    pad_idx = vocab_tgt["<blank>"]
    d_model = 512
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.cuda(gpu)
    module = model
    is_main_process = True
    if is_distributed:
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        )
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    criterion = LabelSmoothing(
        size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1
    )
    criterion.cuda(gpu)

    train_dataloader, valid_dataloader = create_dataloaders(
        gpu,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=config["batch_size"] // ngpus_per_node,
        max_padding=config["max_padding"],
        is_distributed=is_distributed,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup=config["warmup"]
        ),
    )
    train_state = TrainState()

    for epoch in range(config["num_epochs"]):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        print(f"[GPU{gpu}] Эпоха {epoch} Обучение ====", flush=True)
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        GPUtil.showUtilization()
        if is_main_process:
            file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"[GPU{gpu}] Эпоха {epoch} Валидация ====", flush=True)
        model.eval()
        sloss = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        print(sloss)
        torch.cuda.empty_cache()

    if is_main_process:
        file_path = "%sfinal.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path)

def train_distributed_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    from the_annotated_transformer import train_worker

    ngpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    print(f"Количество обнаруженных GPU: {ngpus}")
    print("Запуск процессов обучения ...")
    mp.spawn(
        train_worker,
        nprocs=ngpus,
        args=(ngpus, vocab_src, vocab_tgt, spacy_de, spacy_en, config, True),
    )

def train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    if config["distributed"]:
        train_distributed_model(
            vocab_src, vocab_tgt, spacy_de, spacy_en, config
        )
    else:
        train_worker(
            0, 1, vocab_src, vocab_tgt, spacy_de, spacy_en, config, False
        )

def load_trained_model():
    config = {
        "batch_size": 32,
        "distributed": False,
        "num_epochs": 8,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "file_prefix": "multi30k_model_",
    }
    model_path = "multi30k_model_final.pt"
    if not exists(model_path):
        train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)

    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(torch.load("multi30k_model_final.pt"))
    return model

if is_interactive_notebook():
    model = load_trained_model()
```

> После обучения мы можем декодировать модель для получения набора переводов. Здесь мы просто переводим первое предложение из валидационного набора. Этот набор данных довольно мал, поэтому переводы с жадным поиском достаточно точны.

# Дополнительные компоненты: BPE, поиск, усреднение

> В основном это охватывает саму модель Transformer. Есть четыре аспекта, которые мы не рассмотрели явно. Все эти дополнительные функции реализованы в [OpenNMT-py](https://github.com/opennmt/opennmt-py).

> 1. BPE/Word-piece: Мы можем использовать библиотеку для предварительной обработки данных в подслова. См. реализацию [subword-nmt](https://github.com/rsennrich/subword-nmt) Рико Сеннриха. Эти модели преобразуют обучающие данные в следующий вид:

```
▁Die ▁Protokoll datei ▁kann ▁ heimlich ▁per ▁E - Mail ▁oder ▁FTP ▁an ▁einen ▁bestimmte n ▁Empfänger ▁gesendet ▁werden .
```

> 2. Общие векторные представления: При использовании BPE с общим словарем мы можем разделять одни и те же весовые векторы между источником/целью/генератором. См. [(ссылка)](https://arxiv.org/abs/1608.05859) для деталей. Чтобы добавить это в модель, просто сделайте следующее:

```python
if False:
    model.src_embed[0].lut.weight = model.tgt_embeddings[0].lut.weight
    model.generator.lut.weight = model.tgt_embed[0].lut.weight
```

> 3. Лучевой поиск (Beam Search): Это немного сложно для рассмотрения здесь. См. [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py/) для реализации на pytorch.

> 4. Усреднение модели: В статье усредняются последние k контрольных точек для создания эффекта ансамбля. Мы можем сделать это после факта, если у нас есть несколько моделей:

```python
def average(model, models):
    "Усредняет модели в model"
    for ps in zip(*[m.params() for m in [model] + models]):
        ps[0].copy_(torch.sum(*ps[1:]) / len(ps[1:]))
```

# Результаты

На задаче перевода WMT 2014 английский-немецкий большая модель Transformer (Transformer (big) в Таблице 2) превосходит лучшие ранее опубликованные модели (включая ансамбли) более чем на 2.0 BLEU, устанавливая новый BLEU-рекорд 28.4. Конфигурация этой модели приведена в нижней строке Таблицы 3. Обучение заняло 3.5 дня на 8 GPU P100. Даже наша базовая модель превосходит все ранее опубликованные модели и ансамбли при значительно меньших затратах на обучение по сравнению с любой из конкурирующих моделей.

На задаче перевода WMT 2014 английский-французский наша большая модель достигает BLEU-оценки 41.0, превосходя все ранее опубликованные одиночные модели при менее чем 1/4 затрат на обучение предыдущей лучшей модели. Модель Transformer (big), обученная для английского-французского, использовала коэффициент dropout Pdrop = 0.1 вместо 0.3.

> С дополнительными расширениями из последнего раздела, репликация OpenNMT-py достигает 26.9 на EN-DE WMT. Здесь я загрузил эти параметры в нашу ре-реализацию.

```python
# Загрузка данных и модели для проверки выходов

def check_outputs(
    valid_dataloader,
    model,
    vocab_src,
    vocab_tgt,
    n_examples=15,
    pad_idx=2,
    eos_string="</s>",
):
    results = [()] * n_examples
    for idx in range(n_examples):
        print("\nПример %d ========\n" % idx)
        b = next(iter(valid_dataloader))
        rb = Batch(b[0], b[1], pad_idx)
        greedy_decode(model, rb.src, rb.src_mask, 64, 0)[0]

        src_tokens = [
            vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx
        ]
        tgt_tokens = [
            vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx
        ]

        print(
            "Исходный текст (вход)        : "
            + " ".join(src_tokens).replace("\n", "")
        )
        print(
            "Целевой текст (эталон) : "
            + " ".join(tgt_tokens).replace("\n", "")
        )
        model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]
        model_txt = (
            " ".join(
                [vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
            ).split(eos_string, 1)[0]
            + eos_string
        )
        print("Выход модели               : " + model_txt.replace("\n", ""))
        results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)
    return results

def run_model_example(n_examples=5):
    global vocab_src, vocab_tgt, spacy_de, spacy_en

    print("Подготовка данных ...")
    _, valid_dataloader = create_dataloaders(
        torch.device("cpu"),
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=1,
        is_distributed=False,
    )

    print("Загрузка обученной модели ...")

    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(
        torch.load("multi30k_model_final.pt", map_location=torch.device("cpu"))
    )

    print("Проверка выходов модели:")
    example_data = check_outputs(
        valid_dataloader, model, vocab_src, vocab_tgt, n_examples=n_examples
    )
    return model, example_data

# execute_example(run_model_example)
```

## Визуализация внимания

> Даже с жадным декодером перевод выглядит довольно хорошо. Мы можем дополнительно визуализировать его, чтобы увидеть, что происходит на каждом слое внимания.

```python
def mtx2df(m, max_row, max_col, row_tokens, col_tokens):
    "преобразует плотную матрицу в датафрейм с индексами строк и столбцов"
    return pd.DataFrame(
        [
            (
                r,
                c,
                float(m[r, c]),
                "%.3d %s"
                % (r, row_tokens[r] if len(row_tokens) > r else "<blank>"),
                "%.3d %s"
                % (c, col_tokens[c] if len(col_tokens) > c else "<blank>"),
            )
            for r in range(m.shape[0])
            for c in range(m.shape[1])
            if r < max_row and c < max_col
        ],
        columns=["row", "column", "value", "row_token", "col_token"],
    )

def attn_map(attn, layer, head, row_tokens, col_tokens, max_dim=30):
    df = mtx2df(
        attn[0, head].data,
        max_dim,
        max_dim,
        row_tokens,
        col_tokens,
    )
    return (
        alt.Chart(data=df)
        .mark_rect()
        .encode(
            x=alt.X("col_token", axis=alt.Axis(title="")),
            y=alt.Y("row_token", axis=alt.Axis(title="")),
            color="value",
            tooltip=["row", "column", "value", "row_token", "col_token"],
        )
        .properties(height=400, width=400)
        .interactive()
    )

def get_encoder(model, layer):
    return model.encoder.layers[layer].self_attn.attn

def get_decoder_self(model, layer):
    return model.decoder.layers[layer].self_attn.attn

def get_decoder_src(model, layer):
    return model.decoder.layers[layer].src_attn.attn

def visualize_layer(model, layer, getter_fn, ntokens, row_tokens, col_tokens):
    attn = getter_fn(model, layer)
    n_heads = attn.shape[1]
    charts = [
        attn_map(
            attn,
            0,
            h,
            row_tokens=row_tokens,
            col_tokens=col_tokens,
            max_dim=ntokens,
        )
        for h in range(n_heads)
    ]
    assert n_heads == 8
    return alt.vconcat(
        charts[0]
        # | charts[1]
        | charts[2]
        # | charts[3]
        | charts[4]
        # | charts[5]
        | charts[6]
        # | charts[7]
    ).properties(title="Слой %d" % (layer + 1))
```

## Самовнимание кодера

```python
def viz_encoder_self():
    model, example_data = run_model_example(n_examples=1)
    example = example_data[
        len(example_data) - 1
    ]

    layer_viz = [
        visualize_layer(
            model, layer, get_encoder, len(example[1]), example[1], example[1]
        )
        for layer in range(6)
    ]
    return alt.hconcat(
        layer_viz[0]
        # & layer_viz[1]
        & layer_viz[2]
        # & layer_viz[3]
        & layer_viz[4]
        # & layer_viz[5]
    )

show_example(viz_encoder_self)
```

```
Preparing Data ...
Loading Trained Model ...
Checking Model Outputs:

Example 0 ========

Source Text (Input)        : <s> Zwei Frauen in pinkfarbenen T-Shirts und <unk> unterhalten sich vor einem <unk> . </s>
Target Text (Ground Truth) : <s> Two women wearing pink T - shirts and blue jeans converse outside clothing store . </s>
Model Output               : <s> Two women in pink shirts and face are talking in front of a <unk> . </s>
```

## Самовнимание декодера

```python
def viz_decoder_self():
    model, example_data = run_model_example(n_examples=1)
    example = example_data[len(example_data) - 1]

    layer_viz = [
        visualize_layer(
            model,
            layer,
            get_decoder_self,
            len(example[1]),
            example[1],
            example[1],
        )
        for layer in range(6)
    ]
    return alt.hconcat(
        layer_viz[0]
        & layer_viz[1]
        & layer_viz[2]
        & layer_viz[3]
        & layer_viz[4]
        & layer_viz[5]
    )

show_example(viz_decoder_self)
```

```
Preparing Data ...
Loading Trained Model ...
Checking Model Outputs:

Example 0 ========

Source Text (Input)        : <s> Eine Gruppe von Männern in Kostümen spielt Musik . </s>
Target Text (Ground Truth) : <s> A group of men in costume play music . </s>
Model Output               : <s> A group of men in costumes playing music . </s>
```

## Внимание декодера к источнику

```python
def viz_decoder_src():
    model, example_data = run_model_example(n_examples=1)
    example = example_data[len(example_data) - 1]

    layer_viz = [
        visualize_layer(
            model,
            layer,
            get_decoder_src,
            max(len(example[1]), len(example[2])),
            example[1],
            example[2],
        )
        for layer in range(6)
    ]
    return alt.hconcat(
        layer_viz[0]
        & layer_viz[1]
        & layer_viz[2]
        & layer_viz[3]
        & layer_viz[4]
        & layer_viz[5]
    )

show_example(viz_decoder_src)
```

```
Preparing Data ...
Loading Trained Model ...
Checking Model Outputs:

Example 0 ========

Source Text (Input)        : <s> Ein kleiner Junge verwendet einen Bohrer , um ein Loch in ein Holzstück zu machen . </s>
Target Text (Ground Truth) : <s> A little boy using a drill to make a hole in a piece of wood . </s>
Model Output               : <s> A little boy uses a machine to be working in a hole in a log . </s>
```

# Заключение

Надеемся, этот код будет полезен для будущих исследований. Пожалуйста, свяжитесь с нами, если у вас возникнут какие-либо проблемы.

С уважением, Sasha Rush, Austin Huang, Suraj Subramanian, Jonathan Sum, Khalid Almubarak, Stella Biderman
