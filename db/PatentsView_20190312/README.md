# PatentsView Database Export
This data export contains SQL files that will create the PatentsView database.
## Requirements
* MySQL Database (v5.7 or higher)  
* (Optional but recommended) Myloader/mydumper from here: https://github.com/maxbube/mydumper  

## Files
This folder has the following files  
* One mydumper metadata file  
* 2 SQL files for each table in the database ( a schema & a data file)  
* add-index.sql & drop-index.sql file

## Steps in restoring
### Option 1 (recommended)
* Use myloader command (from mydumper) command with this directory path as input (`myloader --help` has other flags that can be used)  
        EG:`myloader -d /path/to/export/  -s PatentsView_20190312 -v 3 -t {no of parallel threads} -h {hostname} -u {username} -a`
### Option 2 
* All the sql files are in valid MySQL format. MySQL `source filename.sql` can be used to restore them

### Speeding up the restoration process
* The add index and drop index sql files can be used to speed up the import process by
	1. Create the database schema from schema files
	2. Use drop index to drop all indexes
	3. Load data using data sql files
	4. Use add index to recreate indexes
* A rough implementation of above step is available at: https://github.com/American-Institutes-for-Research/mysql-dumper-loader 
