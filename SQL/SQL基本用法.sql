--SQL常用语句汇总
--语句之间一定要用分号隔开
--------------------------------查询------------------------------
select name,imdb_rating from movies;--查询多列数据
select distinct genre from movies; --用于返回唯一不同的值。
select * from movies where name like 'Se_en'; --LIKE 操作符用于在 WHERE 子句中搜索列中的指定模式。此处寻找以Se开头 en结尾的name
select * from movies where name like '%man%';--查找name中包含man
select * from movies where name like '_明%';--查找name中第二个字为明的电影( _替代一个字符，%替代一个或多个字符)
select * from movies where year between 1990 and 2000;
select * from movies where year between 1990 and 2000 and genre='comedy';
select * from movies where genre='comedy' or year<1980;
select * from movies order by imdb_rating desc; --按照imdb_rating降序排列
select * from movies order by imdb_rating Asc limit 3;--查找imdb_rating中最小的三个
--------------------------------函数------------------------
select count(*) from fake_apps;--COUNT(*) 函数返回表中的记录数
select price ,count(*) from fake_apps group by price; --返回每种价格的记录数
select sum(downloads) from fake_apps; --SUM 函数返回数值列的总数（总额）
select max(downloads) from fake_apps;
select avg(downloads) from fake_apps; --AVG 函数返回数值列的平均值。
select price,round(avg(downloads),2) from fake_apps group by price; --round 保留指定小数位数
--------------------------------操作-----------------------------
INSERT INTO celebs (id, name, age) VALUES (5, 'zl', 23); --插入一行
insert or ignore into actor values (3,'ED','CHASE','2006-02-15 12:34:33'); --对于表actor批量插入如下数据,如果数据已经存在，请忽略
update celebs set age=22 where id=1;--修改表中的数据 
update student set ssex='女',sage=sage+1 where sname='赵茵';--修改多个变量值
alter table celebs add column twitter_handle text;--添加一列 (用于修改列属性，添加一列，删除一列)
alter table celebs drop column twitter_handle --删除一列
alter table celebs alter column twitter_handle int --修改列的数据类型
delete from celebs where twitter_handle is NULL; --删除行
create table celeb (id integer primary key,name text unique,date_of_birth text not null,date_of_death text default 'Not Applicable');
--创建表,unique设置该列所有值都不重复,not null设置该列非空,default为该列添加默认值
--创建表时常用的数据类型：integer(size)、int(size)、smallint(size)、tinyint(size)  仅容纳整数。在括号内规定数字的最大位数。
--decimal(size,d)、numeric(size,d) 容纳带有小数的数字。 "size" 规定数字的最大位数。"d" 规定小数点右侧的最大位数。
--char(size) 容纳固定长度的字符串（可容纳字母、数字以及特殊字符）在括号中规定字符串的长度。
--varchar(size) 容纳可变长度的字符串（可容纳字母、数字以及特殊的字符）。在括号中规定字符串的最大长度。
--date(yyyymmdd)  容纳日期。
--------------------------------联结-----------------------------
create table artists(id integer primary key,name text); --id后表示变量类型 primary key 表示该变量作为该表的主键，
                                                         --主键必须包含唯一的值，主键列不能包含 NULL 值
select * from albums where artist_id=3; --条件查询语句
select * from albums join artists on albums.artist_id = artists.id;--内联结查询两张表(返回匹配的行)
select * from albums left join artists on albums.artist_id =artists.id; --即使右表中没有匹配，也从左表返回所有的行
select albums.name as 'album',albums.year,artists.name as 'artist' from albums join artists on albums.artist_id =artists.id where albums.year>1980;
--as 对变量重命名
drop table if exists albums; --当表存在时则删除,不存在时则不进行任何操作
create table if not exists albums (id integer primary key,name text,artist_id integer,year integer,FOREIGN KEY(artist_id) REFERENCES artist(id));
--创建表  FOREIGN KEY 设置外键  一个表中的 FOREIGN KEY 指向另一个表中的 PRIMARY KEY
--------------------------------limit用法-----------------------------
select * from Customer LIMIT 10;--检索前10行数据，显示1-10条数据
select * from Customer LIMIT 1,10;--检索从第2行开始，累加10条id记录，共显示id为2....11
select * from Customer limit 5,10;--检索从第6行开始向前加10条数据，共显示id为6,7....15
select * from Customer limit 6,10;--检索从第7行开始向前加10条记录，显示id为7,8...16
--可以接受两个参数，第一个返回记录行的偏移量，第二个参数指定返回记录行的最大数目
--------------------------------高级教程-----------------------------	
explain select * from employees; --获取select * from employees对应的执行计划
select (last_name ||" " || first_name) as Name from employees;  --将employees表的所有员工的last_name和first_name拼接起来作为Name，中间以一个空格区分
