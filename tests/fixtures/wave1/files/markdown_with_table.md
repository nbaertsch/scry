# Error Codes

## HTTP Status Reference

| Code | Name                  | Description                         |
|------|-----------------------|-------------------------------------|
| 200  | OK                    | Request succeeded                   |
| 400  | Bad Request           | Invalid request syntax              |
| 401  | Unauthorized          | Authentication required             |
| 403  | Forbidden             | Access denied                       |
| 404  | Not Found             | Resource not found                  |
| 500  | Internal Server Error | Unexpected server condition         |

Table cells should be indexed per ``retrieval.bm25.index_table_cells``.
Pipe (``|``) characters serve as word separators in the FTS5 tokenizer.

## gRPC Status Codes

| Code | Name              | Description                     |
|------|-------------------|---------------------------------|
| 0    | OK                | Not an error; returned on success |
| 1    | CANCELLED         | Operation was cancelled         |
| 2    | UNKNOWN           | Unknown error                   |
| 14   | UNAVAILABLE       | Service unavailable             |
