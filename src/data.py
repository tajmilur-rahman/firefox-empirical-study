from bugbug import bugzilla, db, bug_features
import json
import db_connection;
# Downland the latest version if the data set if it is not already downloaded
# db.download(bugzilla.BUGS_DB)

# Iterate over all bugs in the dataset
#for bug in bugzilla.get_bugs():
    # This is the same as if you retrieved the bug through Bugzilla REST API:
    # https://bmo.readthedocs.io/en/latest/api/core/v1/bug.html
    #db_connection.save(bug)

#bug = next(bugzilla.get_bugs())
#for bug in bugzilla.get_bugs():
    #print("----- Bug -----")
    #for key, value in bug.items():
        #print(f"{key}: {value}")

count = 0
for bug in bugzilla.get_bugs():
    count += 1

print("Total bugs:", count)

'''bug = next(bugzilla.get_bugs())   # get the first bug
print("Number of fields:", len(bug))
print("Fields:")
for key in bug.keys():
    print(key)'''

#print("comments", bug["comments"])
#print("cf_status_thunderbird_esr115", bug["cf_status_thunderbird_esr115"]) #The data that can be inputed for product column, so it shows how many rows needed
