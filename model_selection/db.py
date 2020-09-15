#!/usr/bin/env python
# coding: utf-8

# ## Build MySQL database from USPTO Patentsview API database files
# This notebook describes steps to build a MySQL database from the dump files of the USPTO PatentsView API database

# ### Acquire files

# In[ ]:


# create directory for database files
get_ipython().system('mkdir db')


# In[ ]:


# download zippped MySQL dump files of the USPTO PatentsView database
get_ipython().system('wget http://data.patentsview.org/20190312/export/PatentsView_20190312.zip')


# In[ ]:


# unzip files
get_ipython().system('unzip PatentsView_20190312.zip')


# In[ ]:


# change into directory containing unzipped files
get_ipython().system('cd PatentsView_20190312')


# In[ ]:


# inspect number of unzipped files
ls | wc -l
# there are 1412 files that have been unzipeed


# In[ ]:


# inspect type of unzipped files
# use vim or your preferred code editor to open README.md from unzipped files
get_ipython().system(' vim README.md')

# files include: 1 mydumper metadata file, 
#                2 SQL files for each table in the database ( a schema & a data file)
#                1 add-index.sql & 1 drop-index.sql file


# ### Install MySQL

# In[ ]:


# documentation at https://dev.mysql.com/doc/mysql-getting-started/en/


# In[ ]:


# According to documentation, we first add the MySQL APT repository to system's software repository list. 

# Go to the download page for the MySQL APT repository at https://dev.mysql.com/downloads/repo/apt/.

# Select and download the release package for your Linux distribution.

# Install the downloaded release package with the following command, 
# replacing version-specific-package-name with the name of the downloaded package 
# (preceded by its path, if you are not running the command inside the folder where the package is):


# In[ ]:


# download
# for Ubuntu
https://dev.mysql.com/get/mysql-apt-config_0.8.13-1_all.deb


# In[ ]:


# This is after, the mysql menu asking about installation, but there is a prior step not documented here
Which MySQL product do you wish to configure?                             │ 
 │                                                                           │ 
 │          MySQL Server & Cluster (Currently selected: mysql-8.0)           │ 
 │          MySQL Tools & Connectors (Currently selected: Enabled)           │ 
 │          MySQL Preview Packages (Currently selected: Disabled)            │ 
 │          Ok


# In[ ]:


┌─────────────────────┤ Configuring mysql-apt-config ├──────────────────────┐
 │ This configuration program has determined that no MySQL Server is         │ 
 │ configured on your system, and has highlighted the most appropriate       │ 
 │ repository package. If you are not sure which version to install, do not  │ 
 │ change the auto-selected version. Advanced users can always change the    │ 
 │ version as needed later. Note that MySQL Cluster also contains MySQL      │ 
 │ Server.                                                                   │ 
 │                                                                           │ 
 │ Which server version do you wish to receive?                              │ 
 │                                                                           │ 
 │                            mysql-5.7                                      │ 
 │                            mysql-8.0                                      │ 
 │                            mysql-cluster-7.5                              │ 
 │                            mysql-cluster-7.6                              │ 
 │                            mysql-cluster-8.0                              │ 
 │                            None


# In[ ]:


Configuring mysql-apt-config ├──────────────────────┐
 │ MySQL APT Repo features MySQL Server along with a variety of MySQL        │ 
 │ components. You may select the appropriate product to choose the version  │ 
 │ that you wish to receive.                                                 │ 
 │                                                                           │ 
 │ Once you are satisfied with the configuration then select last option     │ 
 │ 'Ok' to save the configuration, then run 'apt-get update' to load         │ 
 │ package list. Advanced users can always change the configurations later,  │ 
 │ depending on their own needs.                                             │ 
 │                                                                           │ 
 │ Which MySQL product do you wish to configure?                             │ 
 │                                                                           │ 
 │          MySQL Server & Cluster (Currently selected: mysql-8.0)           │ 
 │          MySQL Tools & Connectors (Currently selected: Enabled)           │ 
 │          MySQL Preview Packages (Currently selected: Disabled)            │ 
 │          Ok                      


# In[ ]:





# In[ ]:





# In[ ]:


# Asked to set password
Configuring mysql-community-server ├───────────────────┐
 │                                                                           │ 
 │ MySQL 8 uses a new authentication based on improved SHA256-based            
 │ password methods. It is recommended that all new MySQL Server               
 │ installations use this method going forward. This new authentication        
 │ plugin requires new versions of connectors and clients, with support for    
 │ this new authentication method (caching_sha2_password). Currently MySQL     
 │ 8 Connectors and community drivers built with libmysqlclient21 support      
 │ this new method. Clients built with older versions of libmysqlclient may    
 │ not be able to connect to the new server.                                   
 │                                                                             
 │ To retain compatibility with older client software, the default             
 │ authentication plugin can be set to the legacy value                        
 │ (mysql_native_password) This should only be done if required third-party    
 │ software has not been updated to work with the new authentication           
 │ method. The change will be written to the file                              
 │                                                                             
 │                                  <Ok>    


# In[ ]:


Configuring mysql-community-server ├───────────────────┐
 │ Select default authentication plugin                                       │
 │                                                                            │
 │     Use Strong Password Encryption (RECOMMENDED)                           │
 │     Use Legacy Authentication Method (Retain MySQL 5.x Compatibility)      │
 │                                                                            │
 │                                                                            │
 │                                   <Ok>          


# In[ ]:


# Add user accounts
# 6.2.8 Adding Accounts, Assigning Privileges, and Dropping Accounts


# In[ ]:


get_ipython().system('sudo service mysql status')


# ### Build MySQL database
# Per the recommendation in the README.md, we will build the database using the mydumper
# tool, which accelerates the build process through multi-threading

# #### Install mydumper module

# In[ ]:


# the mydumper/myloader module is available here: https://github.com/maxbube/mydumper


# In[ ]:


# download my dumper from https://github.com/maxbube/mydumper
# if on linux:
# check version
get_ipython().system(' wget https://github.com/maxbube/mydumper/releases/download/v0.9.5/mydumper_0.9.5-1.xenial_amd64.deb')


# In[ ]:


# install using apt
get_ipython().system(' sudo apt install mydumper')


# #### Install tmux

# In[ ]:


# we install tmux, a tool that will allow mydumper to run processes in a terminal session
# in the background, even if we end ssh session
# for more info, see https://github.com/tmux/tmux/wiki


# In[ ]:


# install tmux
sudo apt-get install tmux


# In[ ]:


# initiate tmux session in which we run mydumper
tmux


# #### Restore database

# In[ ]:


## Steps in restoring
### Option 1 (recommended)
* Use myloader command (from mydumper) command with this directory path as input (`myloader --help` has other flags that can be used)
        EG:`myloader -d /path/to/export/  -s PatentsView_20190312 -v 3 -t {no of parallel threads} -h {hostname} -u {username} -a`


# In[ ]:


myloader -d /path/to/export/  -s PatentsView_20190312 -v 3 -t 4 -h ip-172-31-8-196 -u ubuntu -a


# In[ ]:


dpkg mydumper-0.9.5-2.el6.x86_64.rpm


# In[ ]:


# install development versions of required libaries (MySQL, GLib, ZLib, PCRE): 
4
# for Ubuntu or Debian: 
get_ipython().system('apt-get install libglib2.0-dev libmysqlclient15-dev zlib1g-dev libpcre3-dev libssl-dev')


# In[ ]:


(base) ubuntu@ip-172-31-8-196:~$ apt-get install libglib2.0-dev libmysqlclient15-dev zlib1g-dev libpcre3-dev libssl-dev
E: Could not open lock file /var/lib/dpkg/lock-frontend - open (13: Permission denied)
get_ipython().set_next_input('E: Unable to acquire the dpkg frontend lock (/var/lib/dpkg/lock-frontend), are you root');get_ipython().run_line_magic('pinfo', 'root')
(base) ubuntu@ip-172-31-8-196:~$ sudo apt-get install libglib2.0-dev libmysqlclient15-dev zlib1g-dev libpcre3-dev libssl-dev


# In[ ]:




