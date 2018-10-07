1）目录操作：ls、cd、mkdir、find、locate、whereis等
--------------------
ls: 查看所有文件夹  
cd：到达指定目录  
mkdir：创建目录  
find:  指定目录搜索（默认当前目录），可以使用文件的属性和权限进行搜索  
locate: 快速查找系统数据库中指定的内容。  
whereis: 速度比较快，因为它只搜索几个特定的目录。   

2）文件操作：mv、cp、rm、touch、cat、more、less
-------------
mv: 移动文件，更改文件名  
　　-i: 交互模式，如果目标文件存在，则询问是否覆盖  
　　-r: 移动目录，跟改目录名  
cp: 拷贝文件 目录  
　　-i: 交互模式，如果目标文件存在，则询问是否覆盖  
　　-r: 拷贝目录  
　　cp file1 file2 file3.... dir 表示将file1,file2...拷贝到dir  
　　cp -r dir1 dir2 dir3... dirn 将dir1, dir2,dir3...拷贝到dirn  
rm: 删除文件  
　　-i: 交互模式，询问是否删除  
　　rm -r dir1 dir2 dir3...可删除多个  
cat: 显示指定文件的全部内容  
more: 分页显示指定文件内容  
Less：用分页的形式显示指定文件的内容，区别是more和less翻页使用的操作键不同。  

3）权限操作：chmod+rwx421

4）账号操作：su、whoami、last、who、w、id、groups等

5）查看系统：history、top

6）关机重启：shutdown、reboot

7）vim操作：i、w、w!、q、q!、wq等

