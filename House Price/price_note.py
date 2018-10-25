    fig, ax = plt.subplots()
    ax.scatter(x=train['GrLivArea'], y=train['SalePrice'])
    plt.ylabel('Price')
    plt.xlabel('GrLivArea')
    plt.show()



    sns.distplot(train['SalePrice'], fit=norm)

    (mu, sigma) = norm.fit(train['SalePrice'])
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
    plt.ylabel('Frequency')
    plt.title('SalePrice distribution')

    fig = plt.figure()
    res = stats.probplot(train['SalePrice'], plot=plt)
    plt.show()


    all_data_na = (all_data.isnull().sum()/len(all_data)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
    print(missing_data.head())

    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    print("\nSkew in numerical features: \n")
    skewness = pd.DataFrame({'Skew' :skewed_feats})
    print(skewness.head(10))
