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
Tar：用于多个文件或目录进行打包，但不压缩，同时也用命令进行解包  
Zip/unzip:压缩解压缩.zip文件  
wc: 计数　　
　　-l: 行数　　
　　-w: 字数　　
　　-c: 字符数　　
　　wc -l file1 file2 ......可以统计多个文件　　
3）权限操作：chmod+rwx421
---------
4）账号操作：su、whoami、last、who、w、id、groups等
-----------
Su：切换用户命令  
Sudo：一系统管理员的身份执行命令  
Passwd：用于修改用户的密码  

5）查看系统：history、top
------------
6）关机重启：shutdown、reboot
　Shutdown：-r 关机后立即重启
　　　　　　-k 并不真正的关机，而只是发出警告信息给所有用户
           -h 关机后不重新启动  
   Reboot： 用于计算机重启
7）vim操作：i、w、w!、q、q!、wq等

8) 其他
------------
man 察看命令详解,只要觉得哪个命令不清楚，man它就可以了.
