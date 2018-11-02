### 基本概念
1. 主键：数据表对数据对象（每一行）有唯一标识和完整的属性或字段   
   外键：数据表中为另一张表的主键的字段
2. drop 删除表  delete删除表中数据，可以用where语句
3. sql创建临时表 create table *** as ; mysql创建临时表 create temporary table ***    
   临时表只在当前会话中存在，当前会话结束后系统自动删除
4. 内连接与外连接
- inner join:返回两张表匹配的行
- left join: 返回左表所有行，以及右表匹配的行
- right join：返回右表所有行，以及左表匹配的行
- full join: 返回两张表所有行
5. 通配符：like "%" 匹配任意字符任意次数（包含0次），‘_’ 匹配单个字符
6. having和where的区别：having 在group by对分组后的数据筛选，而where在分组之前过滤数据
7. 窗口函数 
- 窗口函数的语法: <窗口函数> over ([partition by 列清单] order by <排序用列清单>)
- sum,avg,count,max,min,rank,dense_rank,row_number等聚合函数都可用于窗口函数
- row_number,rank,dense_rank的区别: row_number排序时不会生成相同名次，即没有重复值的排序，
　而rank,dense_rank对于取值相同的样本会赋予同一名次，但rank会跳过之后的名次，而dense_rank不会跳过
```
select
user_id,user_type,sales,
RANK() over (partition by user_type order by sales desc) as r,
ROW_NUMBER() over (partition by user_type order by sales desc) as rn,
DENSE_RANK() over (partition by user_type order by sales desc) as dr
from
orders;
```
上述查询结果如下

| user_id | user_type | sales |  r  | rn  | dr  |
| ------- | --------- | ----- | --- | --- | --- |
| tom1    | new       | 6     | 1   | 1   | 1   |
| tom3    | new       | 5     | 2   | 2   | 2   |
| tom2    | new       | 5     | 2   | 3   | 2   |
| wanger  | new       | 3     | 4   | 4   | 3   |
| zhangsa | new       | 2     | 5   | 5   | 4   |
| tom     | new       | 1     | 6   | 6   | 5   |
| liliu   | new       | 1     | 6   | 7   | 5   |
| tomson  | old       | 3     | 1   | 1   | 1   |
| tomas   | old       | 2     | 2   | 2   | 2   |
| lisi    | old       | 1     | 3   | 3   | 3   |

参考[Hive分析函数和窗口函数](https://www.jianshu.com/p/acc8b158daef)、[窗口函数](https://www.jianshu.com/p/679fd81f8d27)

8. sql语句中根据原始数据生成新字段，例：
```
一张表数据如下
 1900-1-1 胜
 1900-1-1 胜
 1900-1-1 负
 1900-1-2 胜
 1900-1-2 胜
 写出一条SQL语句，使检索结果如下:
          胜  负
 1900-1-1 2   1
 1900-1-2 2   0 
 ```
 采用case when 求解：
 ```
select distinct Date,
sum(case Result when '胜' then 1 else 0 end) as '胜',
sum(case Result when '负' then 1 else 0 end) as '负'
from test
group by date
 
 ```
