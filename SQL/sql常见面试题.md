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
7.窗口函数 除了用于排序还有其他的
