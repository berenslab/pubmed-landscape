import pandas as pd
import xml.etree.ElementTree as et
import os
import torch

def xml_import(xml_file):
    """Parses some elements of the metadata in PubMed XML files.
    Parses the following elements of each paper stored in the input `xml_file`, and stores them in a pandas 
    DataFrame. Elements parsed: PMID, Title, Abstract, Language, Journal, Date, First author name, Last authors name, ISSN.
    
    Parameters
    ----------
    xml_file : path/filename
        Filename of the XML file.
    
    Returns
    -------
    out : pandas dataframe of shape (n_papers, n_elements)
        Pandas dataframe containing all the parsed papers, and as columns the elements of the metadate that were parsed.
    dicc : dict
        Dictionary from which the pandas dataframe was created
    
    See Also
    --------
    xml_import
    
    Notes
    -----
    Parsed elements of each paper:
    - PMID (stored in <PMID>).
    - Title (stored in <ArticleTitle>).
    - Abstract (stored in <AbstractText>).
    - Language (stored in <Language>).
    - Journal (stored in <Title>).
    - Date (stored in <PubDate>).
    - First author first name (stored in <ForeName>, child of <AuthorList>, child of <Author>).
    - Last authors first name (stored in <ForeName>, child of <AuthorList>, child of <Author>).
    - ISSN (stored in <ISSN>)
    
    Details about information extracion: 
    - PMID: If there is no tag <PMID>, it will add 'no tag'. If there is more than one <PMID>,
    it will import only the first one. If <PMID> contains no text, it will add '' (empty string).
    
    - Title: If there is no tag <ArticleTitle>, it will add 'no tag'. If there is more than one
    <ArticleTitle>, it will import only the first one. If <ArticleTitle> contains no text, it will add '' (empty string).
    
    - Abstract: If there is no tag <Abstract> (parent of <AbstractText>), it will add '' (empty string).
    If there is more than one <AbstractText> inside <Abstract>, it will combine them into one list.
    If <AbstractText> contains no text, it will add '' (empty string).
    If there is more than one <Abstract> or other tags containing <AbstractText>, like 
    <OtherAbstract>, it will not get text from them. I am not sure but I think that with the fix it collected all the child tags from <Abstract>.
    
    - Language: If there is no tag <Language>, it will add 'no tag'. If there is more than one
    <Language>, it will import only the first one. If <Language> contains no text, it will add '' (empty string).
    
    - Journal: If there is no tag <Title>, it will add 'no tag'. If there is more than one <Title>,
    it will import only the first one. If <Title> contains no text, it will add '' (empty string).
    
    - Date: If there is no tag <PubDate>, it will add 'no tag'. It will combine all the <PubDate>'s childs' texts
    into one (due to the assymetry of the date storage, sometimes with <Day>, <Month> and <Year>, other times 
    with <MedlineDate>). If <PubDate> contains no further childs, it will print ' '.
    
    - First author first name: It parses the <ForeName> of the first <Author> listed in <Authorlist>. Note that sometimes the metadata is not perfect inside the tag there is the complete name, including surnames. If there is no tag <ForeName>, 'no tag' will be appended. Note for the future: Maybe this is misses some names directly listed in the tag <Author>, maybe an approach similar to what I do for abstracts would be better, where everything under the <Author> tag is parsed. In that case we would also have surnames, but that can be cleaned after.
    
    - Last authors first name: It parses the <ForeName> of the last <Author> listed in <Authorlist>. Note that sometimes the metadata is not perfect inside the tag there is the complete name, including surnames. If there is no tag <ForeName>, 'no tag' will be appended. Note for the future: maybe same problem as in first authors.
    
    - ISSN (stored in <ISSN>): If there is no tag <ISSN>, it will add 'no tag'. If <ISSN> contains no text, it will add '' (empty string). 

    """
    
    
    xtree = et.parse(xml_file, parser=et.XMLParser(encoding="UTF-8"))
    xroot = xtree.getroot()

    dicc={}

    #PMID 
    ros=[]
    for child1 in xroot:
        for child2 in child1:
            for element in child2.iter('MedlineCitation'):
                tag=element.find('PMID')
                if tag is None:
                    ros.append(['no tag'])
                else:
                    res=[]
                    if not tag.text :
                        res.append('')
                    else:
                        res.append(tag.text)
                    ros.append(res)


    ros=[' '.join(ele) for ele in ros]
    dicc['PMID']=ros

    
    #Title 
    ros=[]
    for child1 in xroot:
        for child2 in child1:
            for child3 in child2:
                for article in child3.iter('Article'):
                    tag=article.find('ArticleTitle')
                    if tag is None:
                        ros.append(['no tag'])
                    else:
                        res=[]
                        res.append("".join(tag.find(".").itertext()))
                        ros.append(res)

    ros=[' '.join(ele) for ele in ros]
    dicc['Title']=ros


    #Abstract 
    ros=[]
    for child1 in xroot:
        for child2 in child1:
            for child3 in child2:
                for article in child3.iter('Article'):
                    tag=article.find('Abstract')
                    if tag is None:
                        ros.append([''])
                    else:
                        for child4 in child3:
                            for elem in child4.iter('Abstract'):
                                res=[]
                                for AbstractText in elem.iter('AbstractText'):
                                    res.append("".join(AbstractText.find(".").itertext()).strip())
                                res=[' '.join(res)]
                                res=[elem.strip() for elem in res]
                                ros.append(res)


    #print(ros)
    ros=[' '.join(ele) for ele in ros] 
    dicc['AbstractText']=ros
    
    
    #Language 
    ros=[]
    for child1 in xroot:
        for child2 in child1:
            for child3 in child2:
                for article in child3.iter('Article'):
                    tag=article.find('Language')
                    if tag is None:
                        ros.append(['no tag'])
                    else:
                        res=[]
                        if not tag.text :
                            res.append('')
                        else:
                            res.append(tag.text)
                        ros.append(res)

    ros=[' '.join(ele) for ele in ros]
    dicc['Language']=ros

    
    #Journal 
    ros=[]
    for child1 in xroot:
        for child2 in child1:
            for child3 in child2:
                for child4 in child3:
                    for journal in child4.iter('Journal'):
                        tag=journal.find('Title')
                        if tag is None:
                            ros.append(['no tag'])
                        else:
                            res=[]
                            if not tag.text:
                                res.append('')
                            else:
                                res.append(tag.text)
                            ros.append(res)

    ros=[' '.join(ele) for ele in ros]
    dicc['Journal']=ros

    
    #Date
    ros=[]
    for child1 in xroot:
        for child2 in child1:
            for child3 in child2:
                for child4 in child3:
                    for child5 in child4:
                        for JI in child5.iter('JournalIssue'):
                            tag=JI.find('PubDate')
                            if tag is None:
                                ros.append(['no tag'])
                            else:
                                res=[]
                                for elem in tag:
                                    res.append(elem.text)
                                ros.append(res)
                                
    ros=[' '.join(ele) for ele in ros]
    dicc['Date']=ros
    
    
    #First name of the first author
    ros=[]
    for child1 in xroot:
        for child2 in child1:
            for child3 in child2:
                for child4 in child3.iter('Article'):
                    authorlist = child4.find('AuthorList')
                    if authorlist is None:
                        ros.append('')
                    else:
                        for elem in child4.iter('AuthorList'):
                            author = elem.find('Author')
                            tag=author.find('ForeName')

                            if tag is None:
                                ros.append(['no tag'])
                            else:
                                res=[]
                                if not tag.text:
                                    res.append('')
                                else:
                                    res.append(tag.text)
                                ros.append(res)

    ros=[' '.join(ele) for ele in ros]
    dicc['NameFirstAuthor']=ros
    
    
    #First name of the last author
    ros=[]
    for child1 in xroot:
        for child2 in child1:
            for child3 in child2:
                for child4 in child3.iter('Article'):
                    authorlist = child4.find('AuthorList')
                    if authorlist is None:
                        ros.append('')
                    else:
                        for author in child4.iter('AuthorList'):
                            res=[]
                            for elem in author.iter('Author'):
                                tag=elem.find('ForeName')
                                if tag is None:
                                    res.append(['no tag'])
                                    #for the next time, this is bad because then the element is a list and not a string
                                    #it should be:
                                    # res.append('no tag')  
                                else:
                                    if not tag.text:
                                        res.append('')
                                    else:
                                        res.append(tag.text)
                            ros.append(res[-1])
                            
    dicc['NameLastAuthor']=ros
    
    
    #ISSN 
    ros=[]
    for child1 in xroot:
        for child2 in child1:
            for child3 in child2:
                for child4 in child3:
                    for journal in child4.iter('Journal'):
                        tag=journal.find('ISSN')
                        if tag is None:
                            ros.append(['no tag'])
                        else:
                            res=[]
                            if not tag.text:
                                res.append('')
                            else:
                                res.append(tag.text)
                            ros.append(res)

    ros=[' '.join(ele) for ele in ros]
    dicc['ISSN']=ros

    
    out=pd.DataFrame.from_dict(dicc)
    return out, dicc



def import_all_files(path, order_files=False):
    """Imports all xml files from a directory into a combined dataframe using the function xml_import.
    
    WARNING: I changed the name of the xml_import function that also includes the first names of first and last authors and ISSN, so now this function works calling the old function. A new import_all_files function needs to be created that calls the new xml_import_with_authors_ISSN.
    
    Parameters
    ----------
    path : srt 
        Path of the directory with the files you want to import.
    order_files : bool, default=False 
        If True, it will print the order in which files are being imported.
        
    Returns
    -------
    final_df : pandas dataframe
        Dataframe with all the XML files from the directory imported and merged together (concatenated in the order that they were in the directory (from up to down)).

    """
    # name_files has the names of both .xml files and .gz.md5 files
    name_files=os.listdir(path)
    
    # we select only the .xml files
    len_filenames_list=map(len, name_files)
    len_filenames=np.fromiter(len_filenames_list, dtype=np.int64,count=len(name_files))

    name_files_array=np.array(name_files)
    name_xml_files=name_files_array[len_filenames==17]
    
    # import
    frame_all_df=[]
    
    for i in range(0,len(name_xml_files)):
        path_file=path+name_xml_files[i]
        if order_files==True:
            print(name_xml_files[i])
        df,dic=xml_import(str(path_file))
        dic['filename'] = [name_xml_files[i]]*len(dic['Title'])
        df=pd.DataFrame.from_dict(dic)
        frame_all_df.append(df)

    final_df=pd.concat(frame_all_df,ignore_index=True)
    return final_df



def generate_embeddings(abstracts, tokenizer, model, device):
    """Generate embeddings using BERT-based model.
    Code from Luca Schmidt.

    Parameters
    ----------
    abstracts : list
        Abstract texts.
    tokenizer : transformers.models.bert.tokenization_bert_fast.BertTokenizerFast
        Tokenizer.
    model : transformers.models.bert.modeling_bert.BertModel
        BERT-based model.
        
    Returns
    -------
    embedding_cls : ndarray
        [CLS] tokens of the abstracts.
    embedding_sep : ndarray
        [SEP] tokens of the abstracts.
    embedding_av : ndarray
        Average of tokens of the abstracts.
    """
    # preprocess the input
    inputs = tokenizer(
        abstracts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
    ).to(device)

    # set device
    model = model.to(device)

    # inference
    outputs = model(**inputs)[0].cpu().detach() 

    embedding_av = torch.mean(outputs, [0, 1]).numpy()
    embedding_sep = outputs[:, -1, :].numpy()
    embedding_cls = outputs[:, 0, :].numpy()

    
    return embedding_cls, embedding_sep, embedding_av 