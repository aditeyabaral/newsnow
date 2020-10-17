import utils

print("Welcome to NewsNow!")
choice = 'y'
while choice.lower() == 'y':
    query = "corona virus" #input("Enter query to search: ").lower()
    links = utils.getLinks(query, 5)
    articles = utils.getDocuments(links)
    merged_article = utils.merge(articles)
    print(merged_article)
    exit()
    summary = utils.summarize(merged_article)
    print(f"Latest on {query}:\n")
    print(summary)
    print()
    choice = input("Search for another article? [y/n]: ")
