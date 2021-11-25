 # <center>python学习笔记</center>

## 1.列表的操作

```python
#添加元素
list.append()   
#在索引为1的元素前添加元素a
list.insert(1,'a')
list.pop()
list.count(obj)
list.extend(seq)
list.index(obj)
list.remove(obj)
list.sort()
#对原列表进行反向
list.reverse()
#返回列表元素个数
len(list)
#将元组转为列表
list(sep)
#从start到end，以步长a取元素
list(start:end:a)			 	
#start与end为空时默认start=0,end=len(list)-1以步长为a取元素
list(::a) 
list[-1]表示列表中的最后一个元素，在列表中取元素时步长也可以为负数，但step为-1,start与end为空时，即list(::-1),此时得到的元素相当于将原列表逆序排序
#可以这么理解，当步长为正数时，是从左到右以该步长来获取列表中的元素；
#而当步长为负数时，是从右到左以该步长的绝对值来获取列表中的元素。
```
如果想获取离散的元素，比如想获得第 1、 2、 4 个元素，能不能通过离散的索引值来获取呢？可以用列表推导式
```python
str_list=[x.lower() for x "Lemon"]
list_list=[x for x in [1,23,5]]
tuple_list=[x+2 for x in (1,2,3,4)]
ge_list=[x for x in range(8)]
#两层for循环的列表推导式
two_for_list=[x**2+y for x in range(5) for y in range(4,7)]
```
**使用两个变量来生产list**  
结合字典的使用
```python
d={'x':'1','z':'2'}
d_list=[k+'='+v for k,v in d.item()]
d_list
```
运行结果
```
['x=1','z=2']
```
**含if语句的列表推导式**
```python
if_list=[x**2 for x in range(10) if x%2=0]
```
上述的列表推导式如果用普通的for循环来写的话，代码如下：
```python
if_list=[]
for x in range(10):
    if x%2=0
    if_list.append(x**2)
print(if_list)
```
>通过对比，我们可以看出列表推导式显得更加的pythonic。  
用列表推导式程序运行的时间要比直接用循环结构程序运行的时间要短。  
8. **python中自字典合并的方法**
### Method1:
```python
x={'a':'1','b':'2','c':'3'}
y={'d':'5'}
z={**x,**y}
```
### Method2:自定义函数
```python
def merge(x,y):
    z=x.copy()
    z.update(y)
    return z
merge(x,y)
```
### 多个字典进行合并
```python
#同样可以通过自定义的形式来实现
def merge_dicts(*dict_args):
    result={}
    for item in dict_args:
        result.update(item)
        return result
x1={'x':'23','y':'2'}
x2={'z':'13'}
y1={'d':'34'}
merge_dicts(x1,x2,y1)
```
## 内置时间模块
```mindmap
 Python 内置时间处理模块主要包括：
• Datetime
• Time
• Calendar
```
>概括来说，时间和日期处理的模块和库，有几个主要的概念都是会涉及的，包括：
日期， 时间， 时间戳， 时间和字符串的转换， 时区 等。
**构造时间对象实例** 
### 日期（date）实例的构造
```python
#日期（date）实例的构造
import datetime
d1=datetime.date(2020,10,12
)
d1
```
```
d1=datetime.date(2020,10,12)
```
```python
d1=datetime.date(2020,10,12)
print(d1)
```
运行结果
```
2020-10-12
```
除了上面的构造方式，在date实例的时候，还可以通过date.today
```python
datetime.date.today()#获取今天的日期
```
>date 类型的日期，可以通过 .year , .month , .day 来获取日期所属的年份，月份，和具体的日期号，这几个方
法在数据分析中经常会用到。  
```python
#获取日期所属的年份，月份和具体的日期号

print(f'日期所属的那你年份：{d1.year}')
print(f'日期所属的那你月份：{d1.month}')
print(f'日期具体的日期号：{d1.day}')
```
运行结果：
```python
日期所属的年份：2020
日期所属的月份：10
日期具体的日期号：12
```
### 时间time实例的构造
>time 是一个独立于任何特定日期的理想化时间，其属性有 hour， minute， second ， microsecond 和 tzinfo
。  

```python
t1=datetime.time(20,10,1)
t1
```
```
datetime.time(20,10,1)
```
```python
print(f'time所属的时：{t1.hour}')
print(f'time所属的分 ： {t1.minute}')
print(f'time所属的秒：{t1.second}')
```

### datetime实例的构造
>datetime 是日期和时间的结合，其属性有 year， month， day， hour， minute， second ， microsecond 和
tzinfo 。
```python
dt1=datetime.datetime(2021,4,30,18,44,5)
dt1
```
```
datetime.datetime(20021,4,30,18,44,5)
```
>除了上面的构造方式，在`datetime`实例化的时候，还有其他的一些方式，包括使用 `atetime.now()`和
datetime.today()，以及在 date 的基础上使用 combine 方法等。  
```python
dt2=datetime.datetime.now()
```
```python
dt3=datetime.datetime.today()
```
```python
dt4=datetime.datetime.combine(d1,t1)
```
>通过 `atetime`的实例化，是我们使用时间是经常用到的方法，在日常使用过程中，我们可能只需要具体到某天，
或者只需要具体到某天的某个时间点，这时候，也可以通过 datetime 的一些方法来实现。   
```python
#取日期
dt4.date()
#取具体的时间
dt4.time()
```
>同样的`datetime `型的时间，可以通过 .year , .month , .day 来获取日期所属的年份，月份，和具体的日期
号。
```python
print(f'日期所属的年份：{dt4.year}')
print(f'日期所属的月份：{dt4.month}')
print(f'日期所属的日期号：{dt4.day}')
```
>还有一个可能涉及到的时间是获取某天属于星期几，可以通过 weekday() 和`isoweekday()`方法来实现。  
```python
#从datetime中获取日期为星期几
#使用weekday方法
#数字从0开始，0代表星期一，1代表星期二，以此类推
dt4.weekday()
```
```python
#从datetime来获取日期是星期几
#使用isoweekday方法
#数字从1开始，1代表星期一，2代表星期二，以此类推
dt4.isoweekday()
```
>如果`datetime`的值由于某些原因弄错了，我们也可以通过
replace() 方法来进行更正。这个方法在进行数据清洗的时候会有用。  
>replace 方法的参数如下：  
>`atetime.replace(year=self.year, month=self.month, day=self.day, hour=self.hour,
minute=self.minute, second=self.second, microsecond=self.microsecond, tzinfo=self.
tzinfo, * fold=0)  `
```python
dt5=dt4.replace(year=2021)
#dt4没有改变，只是将dt4的年的部分改为2021以后得到的新值赋给dt5
```
### `imedelta`象的构造
>`imedelta`对象表示两个 date 或者 time 或者`datetime`的时间间隔。  
class datetime.timedelta(days=0, seconds=0, microseconds=0, milliseconds=0,
minutes=0, hours=0, weeks=0)
所有参数都是可选的并且默认为 0。这些参数可以是整数或者浮点数，也可以是正数或者负数。
只有 days, seconds 和 microseconds 会存储在内部。参数单位的换算规则如下：  
• 1 毫秒会转换成 1000 微秒。  
• 1 分钟会转换成 60 秒。  
• 1 小时会转换成 3600 秒。  
• 1 星期会转换成 7 天。  
```python
delta=datetime.timedelta(
    weeks=3,
    day=10,
    hours=6,
    minutes=50,
    seconds=30,
    microseconds=1000,
    milliseconds=10000
)
delta
```
```
datetime.timedelta(31, 24640, 1000)
```
### **`zinfo`绍**
>`atetime.tzinfo `回`datetime `象的时区，前提是在创建`datetime`对象时需传入 `zinfo `数，如果没
有传入则返回值为 None 。  
```python
import pytz
sh=pytz.timezone('Asia/shanghai')
d_tz=datetime.datetime(2020,10,12,tzinfo=sh)
d_tzinfo
```
### 时间转换
>时间的三种存在方式：时间对象，时间字符串，时间戳。  
>时间对象，比如前面介绍的 date 、`datetime`、 time 对象等；时间字符串，如： “2020-10-12”；时间戳，如
time.time() 返回的就是时间戳。  
>在数据处理过程中，经常会遇到需要将不同形式的时间进行转换。 

**1.** **时间对象转字符串**

>间对象转字符串，可以通过`isoformat`或``strftime`来来实现
>strftime的英文全称为str format time,根据给定的样式将时间对象转换为字符串
```python
#将date时间对象转换为字符串
d1=datetime.date(2020,10,12)
d1.isoformat()
```
```python
'2020-10-12'
```
```python
#使用strftime来转换
#YYYY-MM-DD形式
d1.strftime('%Y-%m-%d') #%y得到的年份的后两位
```
```python
'2020-10-12'
```
```python
# MM DD,YYYY形式
d1.strftime('%b %d,%Y')
```
```python
'Oct 12,2020'
```
```python
#将Time时间对象转换为字符串
t1=datetime.time(20,10,1)
t1.strftime('%H:%M:%S')
```
```python
'20:10:01'
```
```python
#将datetime时间对象转换为字符串
dt2=datetime.datetime.now()
dt2.strftime('%Y-%m-%d %H:%M:%S')
```
```python
#运行结果示例：
'2021-04-30 15：12：20'
```
```python
dt2.isoformat()
```
```python
'2021-04-30T15:12:20'
```

>python中常见的时间日期格式化符号：


|指令|意义|示例|
---|---|---
%a|当地工作日的缩写|Sun，Mon,...,Sat(en_US);So,Mo,...,Sa(de_DE)
%A|本地化的星期几完整名称|Sunday,...(en_US),Sonntag,montag,...Samtag(de_DE)
%w|以十进制数显示的工作日，0为周日，6为周六|0，1，2，3，4，5，6
%d|补零后，以十进制数显示的月份中的一天|01,02,...,31
%b|当地月份的缩写|Jan,Feb,...,Dec(en_Us),Jan,Feb,...,Dez(de_DE)
%B|本地化的月份全名|January，February，...，December
%m|补零后以十进制数显示的月份|01,02,...,12
%y|补零后不带世纪的年份|00,01,02,...,99
%Y|带世纪的年份|0001,0002,...,2019,2020,...,9999
%H|补零...小时(24小时制)|00，01，02，23
%I|补零...小时（12小时制）|00,01,...,12
%p|本地化的 AM 或 PM 。|AM, PM (en_US);am, pm (de_DE)
%M|补零后，以十进制数显示的分钟。|00, 01, …, 59
%S|补零后，以十进制数显示的秒。| 00, 01, …, 59
%W |以十进制数表示的一年中的周序号（星期一作为每周的第一天）。在新的一年中第一个第期一之前的所有日子都被视为是在第 0 周。|00, 01, …, 53
%U|以补零后的十进制数表示的一年中的周序号（星期日作为每周的第一天）。在新的一年中第一个星期日之前的所有日子都被视为是在第 0 周。|00, 01, …, 53
%% |字面的 '%' 字符。 |%  
**2.** **字符串转时间对象**
>字符串转时间对象，用的是`strptime` 方法，与 `strftime `方法刚好相反。  
strptime 的英文全称是 str parse time ，将字符串解析为给定相应格式的时间对象。  

```python
s1='2021-05-03'
d=datetime.datetime.strptime(s1,'%Y-%m-%d')
d
```
```python
datetime.datetime(2020,10,9,0,0)
```
>下面提供了 `strftime `方法与` strptime` 方法的比较：

| |strftime|strptime|
---|---|---
用法|根据给定的格式将对象转换为字符串|将字符串解析为给定相应格式的 datetime对象
方法类型|实例方法|类方法
方法|date,datetime,time|datetime
签名|strftime(format)|strptime(date_string,format)
>需要注意的是， strftime 方法可供 date、 time 和 datetime 对象使用，而 strptime 方法仅供 datetime
对象使用。  
**3.****时间戳转换为时间对象**  
>时间戳是指格林威治时间 1970 年 01 月 01 日 00 时 00 分 00 秒 (北京时间 1970 年 01 月 01 日 08 时 00 分 00
秒) 起至现在的总秒数。  
```python
#获取现在的时间戳
ts_1=time.time()
ts_1
```
```python
1619968946.852561
```
```python
#获取当天00：00：00的时间戳
ts_2=int(time.time()/86400)*86400
```
```python
1619913600
```
```python
#获取当天23：59：00的时间戳
#一天有24*60*60=86400秒
ts_3=int(time.time()/86400)*86400+86400-1
```
```python
#将时间戳转换为时间对象
datetime.datetime.fromtimestamp(ts_1)
```
```python
datetime.datetime(2021, 5, 2, 23, 22, 26, 852561)
```
```python
#将时间戳转换为时间对象
datetime.date.fromtimestamp(ts_1)
```
```python
datetime.date(2021, 5, 2)
```
**4.将时间对象转换为时间戳**  
```python
dt1
```
```python
datetime.datetime(2021, 4, 30, 18, 44, 5)
```
```python
#注意这里要用北京时间(东八区)
dt_s=datetime.datetime(1970,1,1,8)
dt_s
```
```python
datetime.datetime(1970, 1, 1, 8, 0)
```
```python
timedelta_1=dt1-dt_s
#返回时间间隔包含了多少秒
timedelta_s=timedelta_1.total_seconds()
timedelta_s
```
```python
1619779445.0
```
>反推一下，是否正确  
```python
#将时间戳转换为时间对象
datetime.datetime.fromtimestamp(timedelta_s)
```
```python
datetime.datetime(2021, 4, 30, 18, 44, 5)
```
### **时间对象的运算**
- **获取当天最小时间和最大时间** 

```python
datetime.datetime.combine(datetime.date.today(),datetime.time.min)
```
```python
datetime.datetime(2021, 5, 2, 0, 0)
```
```python
# 获 取 当 天 最 大 时 间
datetime.datetime.combine(datetime.date.today(),datetime.time.max)
```
```python
datetime.datetime(2021, 5, 2, 23, 59,999999)
```
- **获取当前日期的前几天/后几天**
```python
# 获 取 明 天
datetime.date.today() + datetime.timedelta(days=1)
```
``python
# 获 取 昨 天
datetime.date.today() - datetime.timedelta(days=1)
```
​```python
# 获 取 本 周 第 一 天
d_today = datetime.date.today()
d_today - datetime.timedelta(d_today.weekday())
# 获 取 本 周 最 后 一 天
d_today + datetime.timedelta(6-d_today.weekday())
```
- **计算两个日期相差多少天**
```python
td1 = dt2 - dt1
td1
```
```python
td1.days
```
>如果需要计算两个日期之间总共相差多少秒，应该用 total_seconds() 方法。  
## python时间模块
>time  
>Calendar
```python
#获取某年的日历
import calendar
#获取某年的日历并打印出来
#print calendar
calendar.prcal(2021)
```

![image-20210503142157571](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210503142157571.png)

```python
#获取某月的日历
calendar.month(2021,5)
```

```python
'      May 2021\nMo Tu We Th Fr Sa Su\n                1  2\n 3  4  5  6  7  8  9\n10 11 12 13 14 15 16\n17 18 19 20 21 22 23\n24 25 26 27 28 29 30\n31\n'
```

> 这里需要注意下，在没有使用行数 print() 的情况下，输出的是原始的字符串形式。用 print() 输出后，显示
> 如下：    

```python
      May 2021
Mo Tu We Th Fr Sa Su
                1  2
 3  4  5  6  7  8  9
10 11 12 13 14 15 16
17 18 19 20 21 22 23
24 25 26 27 28 29 30
31
```

> 可以用 prmonth() 函数将结果直接打印出来，效果也是一样的。  

```python
calendar.prmonth(2021,5)
```

- ​	**其他方法**  

```python
#calendar.monthcalendar()
#返回表示一个月的日历的矩阵。每一行代表一周；此月份外的日子由零表示。每周从周一开始，除非使用setfirstweekday() 改变设置
print(calendar.monthcalendar(2020,10))
```

```python
[[0, 0, 0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17, 18],
[19, 20, 21, 22, 23, 24, 25], [26, 27, 28, 29, 30, 31, 0]]
```

```python
#calendar.weekday()
#返回，某天是星期几,默认情况下0-6表示周一到周日
print(calendar.weekday(2020,10,15))
```

```python
3
```

```python
#calendar.setfirstweekday(weekday)
#设置每一周的开始 (0 表示星期一， 6 表示星期天)。 calendar 还提供了 MONDAY, TUESDAY, WEDNESDAY,THURSDAY, FRIDAY, SATURDAY 和 SUNDAY 几个常量方便使用。例如，设置每周的第一天为星期天
import calendar
calendar.serfirstweekday(calendar.SUNDAY)
```

```python
#calendar.weekheader(n)
#返回一个包含星期几的缩写名称，n表示缩写的星期几的字符宽度
print(calendar.weekheader(3))
```

```python
Mon Tue Wed Thu Fri Sat Sun
```
## numpy基本用法  
1. 数组对象的创建  
>numpy 的向量化运算的效率要远远高于 python 的循环遍历运算（效率相差好几
百倍）。  
**基于`list`或`tuple`**  
```python
import numpy as np
arr1=np.aray([1,2,3,4])
arr2=np.array((1,2,3,4))
%两种方法得到的结果是一样的
```
**基于`np.arange`**  
```python
arr1=np.arange(5)
```
**基于arange 以及 reshape 创建多维数组** 
```python
arr=np.arange.reshape(2,3,4)
``` 
*请注意： arange 的长度与 ndarray 的维度的乘积要相等，即 24 = 2X3X4*  
2. ndarray数组的属性  
- `dtype`属性ndarray 数组的数据类型，数据类型的种类，前面已描述。
```python
np.arange(4,dtype=float)
```
```python
array([0.,1.,2.,3.])
```
```python
#'D'表示复数类型
np.arange(4,dtype='D')
```
- ndim 属性，数组维度的数量  
- shape 属性，数组对象的尺度，对于矩阵，即 n 行 m 列,shape 是一个元组（tuple）
- size 属性用来保存元素的数量，相当于 shape 中 nXm 的值
- flat 属性，返回一个 numpy.flatiter 对象，即可迭代的对象。
```python
e=np.arange(6).reshape(2,3)
f=e.flat
for item in f:
    print(f)
```
![]()

[hhh](python学习.ipynb ':include')