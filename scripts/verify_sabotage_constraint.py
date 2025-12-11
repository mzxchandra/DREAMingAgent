
# TF Candidates (from log)
target_tfs = ['b0889', 'b1334', 'b0683', 'b0145', 'b0080', 'b1221', 'b0399', 'b1130', 'b1531', 'b0113', 'b0761', 'b1642', 'b0676', 'b0020', 'b1275', 'b1658', 'b1040', 'b0076', 'b1187', 'b1499', 'b1921', 'b0995', 'b0081', 'b0064', 'b1013', 'b0464', 'b0571', 'b0846', 'b1399', 'b1526', 'b1323', 'b1320', 'b0435', 'b0620', 'b1594', 'b0069', 'b1608', 'b0564', 'b0162', 'b1328', 'b0034', 'b1512', 'b0315', 'b0506', 'b0487', 'b0694', 'b0413', 'b1570', 'b1299', 'b1916', 'b1434', 'b1303', 'b1356', 'b1735', 'b0346', 'b0911', 'b0294', 'b1914', 'b1508', 'b0840', 'b1574', 'b1450', 'b0796', 'b0535', 'b0330', 'b1284', 'b0272', 'b0817', 'b0338', 'b1201', 'b1422', 'b0313', 'b1618', 'b0566', 'b1162', 'b0603', 'b0169', 'b0345', 'b1620', 'b1530', 'b1827', 'b0730', 'b1564', 'b0504', 'b1649', 'b1853', 'b0305', 'b1111', 'b1384', 'b1438', 'b1716', 'b0447', 'b1014', 'b1696', 'b1790', 'b0357', 'b1540', 'b1659', 'b0208', 'b1358']

# Injected False Positives (from log)
injected = ['cynr→hpt', 'rutr→psd', 'fadr→yhaj', 'citr→hflx', 'fimz→hycd', 'cusr→pdeh', 'leuo→leuv', 'narl→spfp', 'cysb→rzpr', 'mcbr→fur', 'mlc→pabb', 'yebk→aroe', 'fliz→glnb', 'racr→yedl', 'rclr→ulae', 'pdhr→btue', 'dhar→cas2', 'phob→fumd', 'envy→ansp', 'deor→alsb', 'fur→yafo', 'rpsa→hlye', 'dicf→exbd', 'racr→rbfa', 'beti→yeha', 'rcda→tord', 'sgrr→uvrb', 'mhpr→arnb', 'ycjw→glpe', 'nhar→yhft', 'uvry→glpd', 'cysb→narh', 'pdel→frlr', 'rpsa→ilvc', 'mode→dicc', 'punr→oxys', 'yebk→hcar', 'comr→rbfa', 'mode→nrfd', 'dicf→thiq', 'kdgr→tort', 'acrr→beta', 'narl→ymda', 'ycit→alav', 'pgrr→ydcc', 'uidr→mlab', 'chbr→mgtl', 'envy→rplp', 'envy→serb', 'slya→yiho']

# Deleted True Edges (from log)
deleted = ['pdhr→mure', 'mcbr→ycif', 'slya→hlye', 'hipb→euta', 'cra→frua', 'fur→gspl', 'slya→sgca', 'pdel→flig', 'phob→rcdb', 'nhar→gltw', 'mata→flhc', 'mata→mata', 'nhar→alau', 'fnr→fnr', 'fur→lysp', 'arac→ygea', 'nhar→rrle', 'phob→phno', 'fur→ybix', 'fur→yihn', 'caif→fixa', 'rutr→gloa', 'dksa→rybb', 'fliz→gadb', 'rclr→rclb', 'phob→phnn', 'ydeo→hyae', 'narl→nark', 'nhar→rsd', 'frmr→frmr', 'cysb→hslj', 'cra→tmar', 'fur→succ', 'slya→cas1', 'ydeo→hyac', 'nagc→chiz', 'fnr→nuon', 'narl→fdhf', 'phob→hiuh', 'cra→seta', 'fnr→hype', 'fnr→nuoj', 'nhar→ileu', 'narl→ynff', 'fur→mnth', 'fur→dmsc', 'mara→pqic', 'lrp→oppf', 'slya→paai', 'fur→rnpb']

# Need to map Names to B-Numbers to check
from src.utils.parsers import parse_gene_product_mapping
name_to_bnumber, _, _ = parse_gene_product_mapping("data/GeneProductAllIdentifiersSet.tsv")

print(f"Checking {len(injected)} Injected Edges and {len(deleted)} Deleted Edges against {len(target_tfs)} Target TFs...")

failed = []

# Check Injected
for edge in injected:
    src, dst = edge.split('→')
    bnum = name_to_bnumber.get(src.lower())
    if bnum not in target_tfs:
        failed.append(f"Injected: {edge} (Source {src} -> {bnum}) NOT in Targets")

# Check Deleted
for edge in deleted:
    src, dst = edge.split('→')
    bnum = name_to_bnumber.get(src.lower())
    if bnum not in target_tfs:
        failed.append(f"Deleted: {edge} (Source {src} -> {bnum}) NOT in Targets")

if not failed:
    print("SUCCESS: All 100 Sabotaged Edges belong to the 100 Target TFs!")
else:
    print(f"FAILURE: {len(failed)} edges failed verification.")
    for f in failed[:10]:
        print(f)
