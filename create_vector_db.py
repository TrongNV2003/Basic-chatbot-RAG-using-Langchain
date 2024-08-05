from preparing.prepare_vector_db import ContextVectorDB, FileVectorDB

context_db = ContextVectorDB()
file_db = FileVectorDB()

# convert a context to vectorDB
# context = "Thép đã tôi thế đấy không phải là một tác phẩm văn học chỉ nhìn đời mà viết. Tác giả sống nó rồi mới viết nó. Nhân vật trung tâm Pa-ven chính là tác giả: Nhi-ca-lai A-xtơ-rốp- xki. Là một chiến sĩ cách mạng tháng Mười, ông đã sống một cách nồng cháy nhất, như nhân vật Pa-ven của ông. Cũng không phải một cuốn tiểu thuyết tự thuật thường vì hứng thú hay lợi ích cá nhân mà viết. A-xtơ-rốp-xki viết Thép đã tôi thế đấy trên giường bệnh, trong khi bại liệt và mù, bệnh tật tàn phá chín phần mười cơ thể. Chưa bao giờ có một nhà văn sáng tác trong những điều kiện gian khổ như vậy. Trong lòng người viết phải có một nhiệt độ cảm hứng nồng nàn không biết bao nhiêu mà kể. Nguồn cảm hứng ấy là sức mạnh tinh thần của người chiến sĩ cách mạng bị tàn phế, đau đớn đến cùng cực, không chịu nằm đợi chết, không thể chịu được xa rời chiến đấu, do đó phấn đấu trở thành một nhà văn và viết nên cuốn sách này. Càng yêu cuốn sách, càng kính trọng nhà văn, càng tôn quí phẩm chất của con người cách mạng."
# context_db.create_db_from_text(context)


# convert all file pdf to vectorDB
pdf_file = "1.pdf"
file_db.create_db_from_files(pdf_file)