import ipaddress
import pandas as pd


# # need to exclude IPv6
# df = pd.read_csv('ip_for_top_acct.csv')
# df = df[df['account_id'] == '1PMs3C13QBace6Xc-DO6wg']
# ip_df = df['xrealip'].unique()
# ip = []
# for i in ip_df:
#     if i.count('.') == 3:
#         ip.append(i)

# cidr = 0

# for i in range(len(ip)):
#     for j in range(len(ip)):
#         ip1_obj=ipaddress.IPv4Address(ip[i])
#         ip2_obj=ipaddress.IPv4Address(ip[j])

#         if ip1_obj<=ip2_obj:
#                 min_ip=ip1_obj
#                 max_ip=ip2_obj
#         else:
#                 min_ip=ip2_obj
#                 max_ip=ip1_obj

#         distance = int(max_ip)-int(min_ip)
#         ip_range=0 #increment powers of 2 until you have subnet distance
#         while 2**ip_range < distance:
#             ip_range += 1

#         net = ipaddress.IPv4Network(str(min_ip) + '/' +str(32-ip_range), strict=False)
#         if max_ip not in net: 
#         # i.e. if the distance implies one size network, but IPs span 2
#             ip_range+=1

#         if ip_range > cidr:
#             cidr = ip_range

# print(32-cidr)





def calc_inclusive_subnet(ip1, ip2): #accepts 2 IP strings
    #make IP Address objects
    ip1_obj=ipaddress.IPv4Address(ip1)
    ip2_obj=ipaddress.IPv4Address(ip2)

    if ip1_obj<=ip2_obj:
        min_ip=ip1_obj
        max_ip=ip2_obj
    else:
        min_ip=ip2_obj
        max_ip=ip1_obj

    distance = int(max_ip)-int(min_ip)
    ip_range=0 #increment powers of 2 until you have subnet distance
    while 2**ip_range < distance:
        ip_range += 1

    net = ipaddress.IPv4Network(str(min_ip) + '/' +str(32-ip_range), strict=False)
    if max_ip not in net: 
    # i.e. if the distance implies one size network, but IPs span 2
        ip_range+=1
        net = ipaddress.IPv4Network(str(min_ip) + '/' +str(32-ip_range), strict=False)

    return net

print(calc_inclusive_subnet('54.241.124.10','54.228.199.150'))






# addresses = []
# # get the int representation of your ip
# ip = int(ipaddress.IPv4Address('100.100.100.105'))

# for mask in range(32):
#     addresses.append(f'{ipaddress.IPv4Address(ip & ((2**32-1)-(2**mask -1)))}/{32-mask }')

# print(addresses)


