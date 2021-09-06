# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
class Table:
    def __init__(self, table=None, 
                columns=['Вопрос', 'Ответ', 'Вероятность'], 
                with_dooble_click_script=True, 
                max_size_rows=None, 
                reverse_data=False):
        if table is None:
            self._table = pd.DataFrame(columns=columns)
        else:
            self._table = table
        self._index_available = 1
        self._columns = columns
        self._table.index = np.arange(1, len(self._table))
        self._max_size_rows = max_size_rows
        self._reverse_data = reverse_data

        if with_dooble_click_script:
            self._html_template = '''
            <head>
                <meta charset="utf-8">
                <link rel="stylesheet" type="text/css" href="../static/style_table.css"/>
            </head>
            <body>
                <center>
                {table}
                </center>
                <script src="../static/doubleClickRowTable.js"></script>
            </body>
            '''
        else:
            self._html_template = '''
            <head>
                <meta charset="utf-8">
                <link rel="stylesheet" type="text/css" href="../static/style_table.css"/>
            </head>
            <body>
                <center>
                {table}
                </center>
            </body>
            '''
            
    def add_row(self, row):
        if self._max_size_rows is not None and self._index_available > self._max_size_rows:
            self._table.drop(index=1, inplace=True)
            self._table.reset_index(drop=True, inplace=True)
            self._table.index = np.arange(1, self._max_size_rows)
            self._table.loc[self._index_available] = row
        else:
            self._table.loc[self._index_available] = row
            self._index_available += 1
    
    def get_html(self, table_id="table_prob", class_table="table_prob"):
        if self._reverse_data:
            table = self._table.iloc[::-1]
            table.reset_index(drop=True, inplace=True)
            table.index = np.arange(1, len(self._table) + 1)
        else:
            table = self._table
        return self._html_template.format(
                    table=table.to_html(
                        classes=class_table, 
                        table_id=table_id,
                        justify="left",
                    )
                )
    
    def save_html(self, path="table.html", table_id="table_prob", class_table="table_prob"):
        with open(path, 'w', encoding="utf-8") as f:
            f.write(self.get_html(table_id=table_id,
                                class_table=class_table,
                                )
            )
    
    def clear_table(self, ):
        del(self._table)
        self._table = pd.DataFrame(columns=self._columns)
        self._table.index = np.arange(1, len(self._table))
        self._index_available = 1

if __name__ == "__main__":
    table = Table()
    table.add_row('Вопрос', 'Ответ', 0.7)
    table.add_row('Ответ', 'Вопрос', 0.7)
    table.add_row('Вопрос', 'Ответ', 0.7)
    table.save_html()