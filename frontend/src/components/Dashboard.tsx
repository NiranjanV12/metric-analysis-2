import {
  DataGrid,
  DataGridComponents,
  Button,
  StatusIndicator,
  Typography,
  Flex,
  Dialog,
  useMediaQuery,
} from '@neo4j-ndl/react';
import { useEffect, useMemo, useState, useCallback, useContext } from 'react';
import { useReactTable, getCoreRowModel, createColumnHelper, getPaginationRowModel } from '@tanstack/react-table';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import { MoonIconOutline, SunIconOutline } from '@neo4j-ndl/react/icons';
import { useNavigate } from 'react-router';
import { IconButtonWithToolTip } from './UI/IconButtonToolTip';
import { MdViewList, MdBugReport } from 'react-icons/md';
import { getServiceHealthAPI, ServiceHealth } from '../services/ServiceHealth';
import { checkIssuesAPI, CheckIssuesResponse } from '../services/CheckIssues';
import { showErrorToast } from '../utils/Toasts';
import { tooltips } from '../utils/Constants';
import { ThemeWrapperContext } from '../context/ThemeWrapper';
import SideNav from './Layout/SideNav';
import DrawerChatbot from './Layout/DrawerChatbot';
import { useMessageContext } from '../context/UserMessages';
import { useCredentials } from '../context/UserCredentials';
import { clearChatAPI } from '../services/QnaAPI';
import { envConnectionAPI } from '../services/ConnectAPI';
import { healthStatus } from '../services/HealthStatus';
import { createDefaultFormData } from '../API/Index';
import { useAuth0 } from '@auth0/auth0-react';
import { Messages } from '../types';

interface HttpStatusData {
  statusCode: number;
  count: number;
}

interface LogLevelData {
  level: string;
  count: number;
}

interface TransactionData {
  type: string;
  count: number;
}

const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const { colorMode, toggleColorMode } = useContext(ThemeWrapperContext);
  const columnHelper = useColumnHelper<ServiceHealth>();
  const isLargeDesktop = useMediaQuery(`(min-width:1440px )`);
  const [isLeftExpanded, setIsLeftExpanded] = useState<boolean>(false);
  const [isRightExpanded, setIsRightExpanded] = useState<boolean>(false);
  const [showDrawerChatbot, setShowDrawerChatbot] = useState<boolean>(true);
  const [chatStarted, setChatStarted] = useState<boolean>(false);
  const {
    connectionStatus,
    setConnectionStatus,
    setUserCredentials,
    setIsBackendConnected,
    setIsGCSActive,
    setGdsActive,
    setIsReadOnlyUser,
    setErrorMessage,
  } = useCredentials();
  const { messages, setMessages, clearHistoryData, setClearHistoryData, setIsDeleteChatLoading } = useMessageContext();
  const { user } = useAuth0();
  const [serviceData, setServiceData] = useState<ServiceHealth[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());

  useEffect(() => {
    async function initializeConnection() {
      try {
        const response = await healthStatus();
        setIsBackendConnected(response.data.healthy);
      } catch (error) {
        setIsBackendConnected(false);
      }
      try {
        const backendApiResponse = await envConnectionAPI();
        const connectionData = backendApiResponse.data;
        if (connectionData.data && connectionData.status === 'Success') {
          const credentials = {
            uri: connectionData.data.uri,
            isReadonlyUser: !connectionData.data.write_access,
            isgdsActive: connectionData.data.gds_status,
            isGCSActive: connectionData.data.gcs_file_cache === 'True',
            chunksTobeProcess: Number(connectionData.data.chunk_to_be_created),
            email: user?.email ?? '',
            connection: 'backendApi',
          };
          setIsGCSActive(credentials.isGCSActive);
          setUserCredentials(credentials);
          createDefaultFormData({ uri: credentials.uri, email: credentials.email ?? '' });
          setGdsActive(credentials.isgdsActive);
          setConnectionStatus(Boolean(connectionData.data.graph_connection));
          setIsReadOnlyUser(connectionData.data.isReadonlyUser);
        } else if (!connectionData.data && connectionData.status === 'Success') {
          const storedCredentials = localStorage.getItem('neo4j.connection');
          if (storedCredentials) {
            const credentials = JSON.parse(storedCredentials);
            setUserCredentials({ ...credentials, password: atob(credentials.password) });
            createDefaultFormData({
              uri: credentials.uri,
              database: credentials.database,
              userName: credentials.userName,
              password: atob(credentials?.password),
              email: credentials.email ?? '',
            });
            setIsGCSActive(credentials.isGCSActive);
            setGdsActive(credentials.isgdsActive);
            setConnectionStatus(Boolean(credentials.connection === 'connectAPI'));
            if (credentials.isReadonlyUser !== undefined) {
              setIsReadOnlyUser(credentials.isReadonlyUser);
            }
          }
        } else {
          setErrorMessage(backendApiResponse?.data?.error);
        }
      } catch (error) {
        if (error instanceof Error) {
          showErrorToast(error.message);
        }
      }
    }
    initializeConnection();
  }, []);

  useEffect(() => {
    if (isRightExpanded && !chatStarted && connectionStatus) {
      setChatStarted(true);
      const callCheckIssues = async () => {
        setIsCheckingIssues(true);
        try {
          const response = await checkIssuesAPI();
          if (response && response.status === 'Success' && response.data) {
            let welcomeMessage = '## System Status Check\n\n';
            if (response.data.display_markdown) {
              welcomeMessage += `${response.data.display_markdown}\n\n`;
            } else if (response.data.diagnosis) {
              welcomeMessage += `**Diagnosis:** ${response.data.diagnosis}\n\n`;
            }
            if (response.data.total_services !== undefined) {
              welcomeMessage += `**Services:** ${response.data.total_services} total, `;
              welcomeMessage += `${response.data.healthy_services} healthy, `;
              welcomeMessage += `${response.data.unhealthy_services} unhealthy\n\n`;
            }
            if (response.data.messages && response.data.messages.length > 0) {
              welcomeMessage += '**AI Analysis:**\n';
              response.data.messages.forEach((msg) => {
                welcomeMessage += `### ${msg.type}\n${msg.content}\n\n`;
              });
            }
            const date = new Date();
            const botMessage: Messages = {
              id: Date.now(),
              user: 'chatbot',
              datetime: date.toLocaleString(),
              isTyping: false,
              isLoading: false,
              modes: {
                [chatModes[0]]: {
                  message: welcomeMessage,
                  sources: response.data.sources ?? [],
                  model: response.data.model ?? '',
                  total_tokens: response.data.total_tokens ?? 0,
                  response_time: response.data.response_time ?? 0,
                  graphonly_entities: response.data.entities ?? [],
                  entities: response.data.entities ?? [],
                  nodeDetails: response.data.nodedetails ?? {},
                  error: undefined,
                },
              },
              currentMode: chatModes[0],
            };
            setMessages((prev: Messages[]) => [...prev, botMessage]);
          }
        } catch (error) {
          console.error('Error checking issues on chat start:', error);
        } finally {
          setIsCheckingIssues(false);
        }
      };
      callCheckIssues();
    }
  }, [isRightExpanded, chatStarted, connectionStatus, setMessages]);

  const chatModes = ['graph_vector_fulltext'];

  const [httpStatusData] = useState<HttpStatusData[]>([
    { statusCode: 200, count: 1450 },
    { statusCode: 201, count: 320 },
    { statusCode: 400, count: 45 },
    { statusCode: 401, count: 12 },
    { statusCode: 403, count: 8 },
    { statusCode: 404, count: 67 },
    { statusCode: 500, count: 5 },
    { statusCode: 502, count: 2 },
    { statusCode: 503, count: 3 },
  ]);

  const [logLevelData] = useState<LogLevelData[]>([
    { level: 'INFO', count: 1250 },
    { level: 'WARN', count: 180 },
    { level: 'ERROR', count: 45 },
    { level: 'DEBUG', count: 320 },
    { level: 'FATAL', count: 3 },
  ]);

  const [transactionData] = useState<TransactionData[]>([
    { type: 'GET', count: 890 },
    { type: 'POST', count: 450 },
    { type: 'PUT', count: 120 },
    { type: 'DELETE', count: 85 },
    { type: 'PATCH', count: 65 },
  ]);

  const [quadrantIssues, setQuadrantIssues] = useState<Record<number, CheckIssuesResponse | null>>({});
  const [activeQuadrant, setActiveQuadrant] = useState<number | null>(null);
  const [isIssuesDialogOpen, setIsIssuesDialogOpen] = useState<boolean>(false);
  const [isCheckingIssues, setIsCheckingIssues] = useState<boolean>(false);

  const toggleLeftDrawer = useCallback(() => {
    if (isLargeDesktop) {
      setIsLeftExpanded((old) => !old);
    } else {
      setIsLeftExpanded(false);
    }
  }, [isLargeDesktop]);

  const toggleRightDrawer = useCallback(() => {
    if (isLargeDesktop) {
      setIsRightExpanded((prev) => !prev);
    } else {
      setIsRightExpanded(false);
    }
  }, [isLargeDesktop]);

  const deleteOnClick = useCallback(async () => {
    try {
      setClearHistoryData(true);
      setIsDeleteChatLoading(true);
      const response = await clearChatAPI(sessionStorage.getItem('session_id') ?? '');
      setIsDeleteChatLoading(false);
      if (response.data.status === 'Success') {
        const date = new Date();
        setMessages([
          {
            datetime: `${date.toLocaleDateString()} ${date.toLocaleTimeString()}`,
            id: 2,
            modes: {
              'graph+vector+fulltext': {
                message:
                  ' Welcome to the Neo4j Knowledge Graph Chat. You can ask questions related to documents which have been completely processed.',
              },
            },
            user: 'chatbot',
            currentMode: 'graph+vector+fulltext',
          },
        ]);
        navigate('.', { replace: true, state: null });
      }
    } catch (error) {
      setIsDeleteChatLoading(false);
      console.log(error);
      setClearHistoryData(false);
    }
  }, []);

  const fetchServiceHealth = useCallback(async () => {
    try {
      const data = await getServiceHealthAPI();
      setServiceData(data);
      setLastUpdated(new Date());
    } catch (error) {
      showErrorToast('Failed to fetch service health data');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchServiceHealth();
    const intervalId = setInterval(() => {
      fetchServiceHealth();
    }, 30000);
    return () => clearInterval(intervalId);
  }, [fetchServiceHealth]);

  const columns = useMemo(
    () => [
      columnHelper.accessor((row) => row.service, {
        id: 'service',
        cell: (info) => (
          <Typography variant='body-medium' className='font-semibold'>
            {info.getValue()}
          </Typography>
        ),
        header: () => <span>Service</span>,
        size: 150,
      }),
      columnHelper.accessor((row) => row.healthUrl, {
        id: 'healthUrl',
        cell: (info) => (
          <Typography variant='body-medium' className='text-palette-primary-text'>
            {info.getValue()}
          </Typography>
        ),
        header: () => <span>Health URL</span>,
        size: 300,
      }),
      columnHelper.accessor((row) => row.status, {
        id: 'status',
        cell: (info) => {
          const status = info.getValue();
          const statusType = status === 'Running' ? 'success' : status === 'Stopped' ? 'danger' : 'warning';
          return (
            <div className='flex items-center gap-2'>
              <StatusIndicator type={statusType} />
              <Typography variant='body-medium'>{status}</Typography>
            </div>
          );
        },
        header: () => <span>Status</span>,
        size: 120,
      }),
    ],
    [columnHelper]
  );

  const table = useReactTable({
    data: serviceData,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
  });

  const handleBackToHome = () => {
    navigate('/');
  };

  const handleCheckIssues = async (quadrant: number) => {
    setActiveQuadrant(quadrant);
    setIsCheckingIssues(true);
    try {
      const response = await checkIssuesAPI();
      if (response && response.status === 'Success') {
        setQuadrantIssues((prev) => ({ ...prev, [quadrant]: response }));
        setIsIssuesDialogOpen(true);
      } else {
        showErrorToast('No issues data returned from agent');
      }
    } catch (error) {
      showErrorToast('Failed to check issues');
    } finally {
      setIsCheckingIssues(false);
    }
  };

  const handleCloseIssuesDialog = () => {
    setIsIssuesDialogOpen(false);
    setActiveQuadrant(null);
  };

  const getMaxCount = (data: { count: number }[]) => Math.max(...data.map((d) => d.count));

  const getBarColor = (index: number) => {
    const colors = [
      'bg-palette-primary-bg-strong',
      'bg-palette-success-bg-strong',
      'bg-palette-warning-bg-strong',
      'bg-palette-danger-bg-strong',
      'bg-palette-neutral-bg-strong',
    ];
    return colors[index % colors.length];
  };

  const renderBarChart = <T extends { count: number }>(data: T[], getLabel: (d: T) => string) => {
    const maxCount = getMaxCount(data);
    return (
      <div className='flex flex-col gap-2 h-full'>
        {data.map((item, index) => (
          <div key={index} className='flex items-center gap-2'>
            <Typography variant='body-small' className='w-16 shrink-0'>
              {getLabel(data[index])}
            </Typography>
            <div className='flex-1 h-6 bg-palette-neutral-bg-weak rounded overflow-hidden'>
              <div
                className={`h-full ${getBarColor(index)} transition-all duration-300`}
                style={{ width: `${(item.count / maxCount) * 100}%` }}
              />
            </div>
            <Typography variant='body-small' className='w-12 text-right'>
              {item.count}
            </Typography>
          </div>
        ))}
      </div>
    );
  };

  const formatLastUpdated = () => lastUpdated.toLocaleTimeString();

  const getQuadrantTitle = (quadrant: number) => {
    const titles: Record<number, string> = {
      0: 'All Issues',
      1: 'HTTP Status Code',
      2: 'Service Overview Status',
      3: 'Log Monitoring',
      4: 'Transaction Monitoring',
    };
    return titles[quadrant];
  };

  const currentAgentResponse = activeQuadrant !== null ? quadrantIssues[activeQuadrant] : null;

  if (isLargeDesktop) {
    return (
      <>
        {isCheckingIssues && (
          <div
            className='fixed inset-0 flex items-center justify-center z-50'
            style={{ backgroundColor: 'rgba(0, 0, 0, 0.5)', backdropFilter: 'blur(4px)' }}
          >
            <div
              className='n-flex n-flex-col n-justify-center n-items-center n-gap-y-2'
              style={{ backgroundColor: 'white', padding: '24px', borderRadius: '8px' }}
            >
              <div className='ndl-spin-wrapper ndl-large' role='status' aria-label='Loading content' aria-live='polite'>
                <div className='ndl-spin'></div>
              </div>
              <Typography variant='body-large' className='n-font-semibold'>
                Checking for issues...
              </Typography>
            </div>
          </div>
        )}
        <div
          className={`layout-wrapper ${!isLeftExpanded ? 'drawerdropzoneclosed' : ''} ${
            !isRightExpanded ? 'drawerchatbotclosed' : ''
          } ${!isRightExpanded && !isLeftExpanded ? 'drawerclosed' : ''}`}
        >
          <SideNav
            toggles3Modal={() => {}}
            toggleGCSModal={() => {}}
            toggleGenericModal={() => {}}
            isExpanded={isLeftExpanded}
            position='left'
            toggleDrawer={toggleLeftDrawer}
          />
          {isLeftExpanded && (
            <div className='flex relative min-h-[calc(-58px+100vh)]'>
              <div className='w-[294px] p-4 bg-palette-neutral-bg-default border-r'>
                <Typography variant='h6' className='mb-4'>
                  Dashboard Options
                </Typography>
                <Typography variant='body-small'>Additional dashboard controls can go here</Typography>
              </div>
            </div>
          )}
          <div className='n-bg-palette-neutral-bg-weak min-h-screen flex flex-col relative'>
            <div
              className='n-bg-palette-neutral-bg-weak p-1'
              style={{ borderBottom: '2px solid rgb(var(--theme-palette-neutral-border-weak))' }}
            >
              <nav className='flex items-center justify-between flex-row' role='navigation'>
                <section className='flex w-1/3 shrink-0 grow-0 items-center min-w-[200px]'></section>
                <section className='flex w-1/3 justify-center'>
                  <Typography variant='h4'>Service Observability Dashboard</Typography>
                </section>
                <section className='items-center justify-end w-1/3 grow-0 flex'>
                  <div
                    className='inline-flex gap-x-1'
                    style={{ display: 'flex', flexGrow: 0, alignItems: 'center', gap: '4px' }}
                  >
                    <IconButtonWithToolTip
                      label={tooltips.theme}
                      text={tooltips.theme}
                      clean
                      size='large'
                      onClick={toggleColorMode}
                      placement='left'
                    >
                      {colorMode === 'dark' ? (
                        <span role='img' aria-label='sun'>
                          <SunIconOutline />
                        </span>
                      ) : (
                        <span role='img' aria-label='moon'>
                          <MoonIconOutline />
                        </span>
                      )}
                    </IconButtonWithToolTip>
                    <IconButtonWithToolTip
                      label='Check for Issue'
                      text='Check for Issue'
                      clean
                      size='large'
                      onClick={() => handleCheckIssues(0)}
                      placement='left'
                      disabled={isCheckingIssues}
                    >
                      <MdBugReport />
                    </IconButtonWithToolTip>
                    <IconButtonWithToolTip
                      label='Detailed View'
                      text='Detailed View'
                      clean
                      size='large'
                      onClick={handleBackToHome}
                      placement='left'
                    >
                      <MdViewList />
                    </IconButtonWithToolTip>
                  </div>
                </section>
              </nav>
            </div>

            <div className='p-4 flex-1'>
              <Typography variant='body-small' className='text-palette-neutral-text-weak mb-4'>
                Last updated: {formatLastUpdated()} | Auto-refresh: 30s
              </Typography>

              <div className='grid grid-cols-2 gap-4 h-[calc(100vh-180px)]' style={{ minWidth: '1200px' }}>
                {/* Quadrant 1: HTTP Status Code Graph */}
                <div className='p-4 flex flex-col border rounded-lg bg-palette-neutral-bg-default'>
                  <div className='flex justify-center items-center mb-4 relative'>
                    <Typography variant='h5'>{getQuadrantTitle(1)}</Typography>
                    <div className='absolute right-0'>
                      <IconButtonWithToolTip
                        label='Check Issues'
                        text='Check Issues'
                        clean
                        size='small'
                        onClick={() => handleCheckIssues(1)}
                        placement='left'
                      >
                        <MdBugReport />
                      </IconButtonWithToolTip>
                    </div>
                  </div>
                  <div className='flex-1 overflow-auto'>{renderBarChart(httpStatusData, (d) => `${d.statusCode}`)}</div>
                </div>

                {/* Quadrant 2: Service Overview Status */}
                <div
                  className='p-4 flex flex-col border rounded-lg bg-palette-neutral-bg-default'
                  style={{ minWidth: '600px' }}
                >
                  <div className='flex justify-center items-center mb-4 relative'>
                    <Typography variant='h5'>{getQuadrantTitle(2)}</Typography>
                    <div className='absolute right-0'>
                      <IconButtonWithToolTip
                        label='Check Issues'
                        text='Check Issues'
                        clean
                        size='small'
                        onClick={() => handleCheckIssues(2)}
                        placement='left'
                      >
                        <MdBugReport />
                      </IconButtonWithToolTip>
                    </div>
                  </div>
                  <div className='flex-1 overflow-auto' style={{ minWidth: '600px' }}>
                    <DataGrid
                      isResizable={true}
                      tableInstance={table}
                      styling={{
                        borderStyle: 'all-sides',
                        hasZebraStriping: true,
                        headerStyle: 'clean',
                      }}
                      isLoading={isLoading}
                      rootProps={{
                        className: 'w-full',
                        style: { width: '100%' },
                      }}
                      components={{
                        TableResults: () => (
                          <DataGridComponents.TableResults>
                            <Typography variant='body-medium'>
                              {serviceData.length === 0 && !isLoading
                                ? 'No services configured'
                                : `${table.getRowModel().rows.length} service(s)`}
                            </Typography>
                          </DataGridComponents.TableResults>
                        ),
                      }}
                    />
                  </div>
                </div>

                {/* Quadrant 3: Log Monitoring */}
                <div className='p-4 flex flex-col border rounded-lg bg-palette-neutral-bg-default'>
                  <div className='flex justify-center items-center mb-4 relative'>
                    <Typography variant='h5'>{getQuadrantTitle(3)}</Typography>
                    <div className='absolute right-0'>
                      <IconButtonWithToolTip
                        label='Check Issues'
                        text='Check Issues'
                        clean
                        size='small'
                        onClick={() => handleCheckIssues(3)}
                        placement='left'
                      >
                        <MdBugReport />
                      </IconButtonWithToolTip>
                    </div>
                  </div>
                  <div className='flex-1 overflow-auto'>{renderBarChart(logLevelData, (d) => d.level)}</div>
                </div>

                {/* Quadrant 4: Transaction Monitoring */}
                <div className='p-4 flex flex-col border rounded-lg bg-palette-neutral-bg-default'>
                  <div className='flex justify-center items-center mb-4 relative'>
                    <Typography variant='h5'>{getQuadrantTitle(4)}</Typography>
                    <div className='absolute right-0'>
                      <IconButtonWithToolTip
                        label='Check Issues'
                        text='Check Issues'
                        clean
                        size='small'
                        onClick={() => handleCheckIssues(4)}
                        placement='left'
                      >
                        <MdBugReport />
                      </IconButtonWithToolTip>
                    </div>
                  </div>
                  <div className='flex-1 overflow-auto'>{renderBarChart(transactionData, (d) => d.type)}</div>
                </div>
              </div>
            </div>

            <Dialog isOpen={isIssuesDialogOpen} onClose={handleCloseIssuesDialog} size='large'>
              <Dialog.Header>
                AI Agent Analysis - {activeQuadrant ? getQuadrantTitle(activeQuadrant) : ''}
              </Dialog.Header>
              <Dialog.Content className='n-flex n-flex-col n-gap-token-4'>
                {currentAgentResponse?.data && (
                  <>
                    <div>
                      <Typography variant='h5' className='mb-2'>
                        Query
                      </Typography>
                      <Typography variant='body-medium'>{currentAgentResponse.data.query}</Typography>
                    </div>

                    {currentAgentResponse.data.display_markdown && (
                      <div>
                        <Typography variant='h5' className='mb-2'>
                          Analysis
                        </Typography>
                        <div className='p-3 bg-palette-neutral-bg-weak rounded'>
                          <Typography variant='body-medium'>{currentAgentResponse.data.display_markdown}</Typography>
                        </div>
                      </div>
                    )}

                    {currentAgentResponse.data.total_services !== undefined && (
                      <div>
                        <Typography variant='h5' className='mb-2'>
                          Service Summary
                        </Typography>
                        <Flex gap='4'>
                          <div className='p-2 border rounded'>
                            <Typography variant='body-small'>Total</Typography>
                            <Typography variant='h4'>{currentAgentResponse.data.total_services}</Typography>
                          </div>
                          <div className='p-2 border rounded bg-palette-success-bg-weak'>
                            <Typography variant='body-small'>Healthy</Typography>
                            <Typography variant='h4' className='text-palette-success-text'>
                              {currentAgentResponse.data.healthy_services}
                            </Typography>
                          </div>
                          <div className='p-2 border rounded bg-palette-danger-bg-weak'>
                            <Typography variant='body-small'>Unhealthy</Typography>
                            <Typography variant='h4' className='text-palette-danger-text'>
                              {currentAgentResponse.data.unhealthy_services}
                            </Typography>
                          </div>
                        </Flex>
                      </div>
                    )}

                    {currentAgentResponse.data.services && currentAgentResponse.data.services.length > 0 && (
                      <div>
                        <Typography variant='h5' className='mb-2'>
                          Services Status
                        </Typography>
                        <div className='space-y-2'>
                          {currentAgentResponse.data.services.map((service, index) => (
                            <div key={index} className='p-2 border rounded flex justify-between'>
                              <Typography variant='body-medium'>{service.service}</Typography>
                              <Flex gap='2' alignItems='center'>
                                <StatusIndicator
                                  type={
                                    service.status === 'Running'
                                      ? 'success'
                                      : service.status === 'Stopped'
                                        ? 'danger'
                                        : 'warning'
                                  }
                                />
                                <Typography variant='body-small'>{service.status}</Typography>
                              </Flex>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {currentAgentResponse.data.messages && currentAgentResponse.data.messages.length > 0 && (
                      <div>
                        <Typography variant='h5' className='mb-2'>
                          Agent Messages
                        </Typography>
                        <div className='space-y-2 max-h-60 overflow-y-auto prose prose-sm'>
                          {currentAgentResponse.data.messages.map((msg, index) => (
                            <div key={index} className='p-2 border rounded bg-palette-neutral-bg-weak'>
                              <Typography variant='body-small' className='font-semibold'>
                                {msg.type}
                              </Typography>
                              <div className='mt-1'>
                                <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw] as any}>
                                  {msg.content}
                                </ReactMarkdown>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </>
                )}
              </Dialog.Content>
              <Dialog.Actions className='mt-4'>
                <Button onClick={handleCloseIssuesDialog}>Close</Button>
              </Dialog.Actions>
            </Dialog>
          </div>
          {isRightExpanded && (
            <DrawerChatbot
              messages={messages}
              isExpanded={isRightExpanded}
              clearHistoryData={clearHistoryData}
              connectionStatus={connectionStatus}
              setMessages={setMessages}
              isDeleteChatLoading={false}
            />
          )}
          <SideNav
            messages={messages}
            isExpanded={isRightExpanded}
            position='right'
            toggleDrawer={toggleRightDrawer}
            deleteOnClick={deleteOnClick}
            showDrawerChatbot={showDrawerChatbot}
            setShowDrawerChatbot={setShowDrawerChatbot}
            setIsRightExpanded={setIsRightExpanded}
            clearHistoryData={clearHistoryData}
            toggleGCSModal={() => {}}
            toggles3Modal={() => {}}
            toggleGenericModal={() => {}}
            setIsleftExpanded={setIsLeftExpanded}
          />
        </div>
      </>
    );
  }

  return (
    <>
      {isCheckingIssues && (
        <div
          className='fixed inset-0 flex items-center justify-center z-50'
          style={{ backgroundColor: 'rgba(0, 0, 0, 0.5)', backdropFilter: 'blur(4px)' }}
        >
          <div
            className='n-flex n-flex-col n-justify-center n-items-center n-gap-y-2'
            style={{ backgroundColor: 'white', padding: '24px', borderRadius: '8px' }}
          >
            <div className='ndl-spin-wrapper ndl-large' role='status' aria-label='Loading content' aria-live='polite'>
              <div className='ndl-spin'></div>
            </div>
            <Typography variant='body-large' className='n-font-semibold'>
              Checking for issues...
            </Typography>
          </div>
        </div>
      )}
      <div className='layout-wrapper drawerclosed'>
        <SideNav
          toggles3Modal={() => {}}
          toggleGCSModal={() => {}}
          toggleGenericModal={() => {}}
          isExpanded={isLeftExpanded}
          position='left'
          toggleDrawer={toggleLeftDrawer}
        />
        <div className='n-bg-palette-neutral-bg-weak min-h-screen flex flex-col relative'>
          <div
            className='n-bg-palette-neutral-bg-weak p-1'
            style={{ borderBottom: '2px solid rgb(var(--theme-palette-neutral-border-weak))' }}
          >
            <nav className='flex items-center justify-between flex-row' role='navigation'>
              <section className='flex w-1/3 shrink-0 grow-0 items-center min-w-[200px]'></section>
              <section className='flex w-1/3 justify-center'>
                <Typography variant='h4'>Service Observability Dashboard</Typography>
              </section>
              <section className='items-center justify-end w-1/3 grow-0 flex'>
                <div
                  className='inline-flex gap-x-1'
                  style={{ display: 'flex', flexGrow: 0, alignItems: 'center', gap: '4px' }}
                >
                  <IconButtonWithToolTip
                    label={tooltips.theme}
                    text={tooltips.theme}
                    clean
                    size='large'
                    onClick={toggleColorMode}
                    placement='left'
                  >
                    {colorMode === 'dark' ? (
                      <span role='img' aria-label='sun'>
                        <SunIconOutline />
                      </span>
                    ) : (
                      <span role='img' aria-label='moon'>
                        <MoonIconOutline />
                      </span>
                    )}
                  </IconButtonWithToolTip>
                  <IconButtonWithToolTip
                    label='Check for Issue'
                    text='Check for Issue'
                    clean
                    size='large'
                    onClick={() => handleCheckIssues(0)}
                    placement='left'
                    disabled={isCheckingIssues}
                  >
                    <MdBugReport />
                  </IconButtonWithToolTip>
                  <IconButtonWithToolTip
                    label='Detailed View'
                    text='Detailed View'
                    clean
                    size='large'
                    onClick={handleBackToHome}
                    placement='left'
                  >
                    <MdViewList />
                  </IconButtonWithToolTip>
                </div>
              </section>
            </nav>
          </div>

          <div className='p-4 flex-1'>
            <Typography variant='body-small' className='text-palette-neutral-text-weak mb-4'>
              Last updated: {formatLastUpdated()} | Auto-refresh: 30s
            </Typography>

            <div className='grid grid-cols-2 gap-4 h-[calc(100vh-180px)]' style={{ minWidth: '1200px' }}>
              {/* Quadrant 1: HTTP Status Code Graph */}
              <div className='p-4 flex flex-col border rounded-lg bg-palette-neutral-bg-default'>
                <div className='flex justify-center items-center mb-4 relative'>
                  <Typography variant='h5'>{getQuadrantTitle(1)}</Typography>
                  <div className='absolute right-0'>
                    <IconButtonWithToolTip
                      label='Check Issues'
                      text='Check Issues'
                      clean
                      size='small'
                      onClick={() => handleCheckIssues(1)}
                      placement='left'
                    >
                      <MdBugReport />
                    </IconButtonWithToolTip>
                  </div>
                </div>
                <div className='flex-1 overflow-auto'>{renderBarChart(httpStatusData, (d) => `${d.statusCode}`)}</div>
              </div>

              {/* Quadrant 2: Service Overview Status */}
              <div
                className='p-4 flex flex-col border rounded-lg bg-palette-neutral-bg-default'
                style={{ minWidth: '600px' }}
              >
                <div className='flex justify-center items-center mb-4 relative'>
                  <Typography variant='h5'>{getQuadrantTitle(2)}</Typography>
                  <div className='absolute right-0'>
                    <IconButtonWithToolTip
                      label='Check Issues'
                      text='Check Issues'
                      clean
                      size='small'
                      onClick={() => handleCheckIssues(2)}
                      placement='left'
                    >
                      <MdBugReport />
                    </IconButtonWithToolTip>
                  </div>
                </div>
                <div className='flex-1 overflow-auto' style={{ minWidth: '600px' }}>
                  <DataGrid
                    isResizable={true}
                    tableInstance={table}
                    styling={{
                      borderStyle: 'all-sides',
                      hasZebraStriping: true,
                      headerStyle: 'clean',
                    }}
                    isLoading={isLoading}
                    rootProps={{
                      className: 'w-full',
                      style: { width: '100%' },
                    }}
                    components={{
                      TableResults: () => (
                        <DataGridComponents.TableResults>
                          <Typography variant='body-medium'>
                            {serviceData.length === 0 && !isLoading
                              ? 'No services configured'
                              : `${table.getRowModel().rows.length} service(s)`}
                          </Typography>
                        </DataGridComponents.TableResults>
                      ),
                    }}
                  />
                </div>
              </div>

              {/* Quadrant 3: Log Monitoring */}
              <div className='p-4 flex flex-col border rounded-lg bg-palette-neutral-bg-default'>
                <div className='flex justify-center items-center mb-4 relative'>
                  <Typography variant='h5'>{getQuadrantTitle(3)}</Typography>
                  <div className='absolute right-0'>
                    <IconButtonWithToolTip
                      label='Check Issues'
                      text='Check Issues'
                      clean
                      size='small'
                      onClick={() => handleCheckIssues(3)}
                      placement='left'
                    >
                      <MdBugReport />
                    </IconButtonWithToolTip>
                  </div>
                </div>
                <div className='flex-1 overflow-auto'>{renderBarChart(logLevelData, (d) => d.level)}</div>
              </div>

              {/* Quadrant 4: Transaction Monitoring */}
              <div className='p-4 flex flex-col border rounded-lg bg-palette-neutral-bg-default'>
                <div className='flex justify-center items-center mb-4 relative'>
                  <Typography variant='h5'>{getQuadrantTitle(4)}</Typography>
                  <div className='absolute right-0'>
                    <IconButtonWithToolTip
                      label='Check Issues'
                      text='Check Issues'
                      clean
                      size='small'
                      onClick={() => handleCheckIssues(4)}
                      placement='left'
                    >
                      <MdBugReport />
                    </IconButtonWithToolTip>
                  </div>
                </div>
                <div className='flex-1 overflow-auto'>{renderBarChart(transactionData, (d) => d.type)}</div>
              </div>
            </div>
          </div>

          <Dialog isOpen={isIssuesDialogOpen} onClose={handleCloseIssuesDialog} size='large'>
            <Dialog.Header>AI Agent Analysis - {activeQuadrant ? getQuadrantTitle(activeQuadrant) : ''}</Dialog.Header>
            <Dialog.Content className='n-flex n-flex-col n-gap-token-4'>
              {currentAgentResponse?.data && (
                <>
                  <div>
                    <Typography variant='h5' className='mb-2'>
                      Query
                    </Typography>
                    <Typography variant='body-medium'>{currentAgentResponse.data.query}</Typography>
                  </div>

                  {currentAgentResponse.data.display_markdown && (
                    <div>
                      <Typography variant='h5' className='mb-2'>
                        Analysis
                      </Typography>
                      <div className='p-3 bg-palette-neutral-bg-weak rounded'>
                        <Typography variant='body-medium'>{currentAgentResponse.data.display_markdown}</Typography>
                      </div>
                    </div>
                  )}

                  {currentAgentResponse.data.total_services !== undefined && (
                    <div>
                      <Typography variant='h5' className='mb-2'>
                        Service Summary
                      </Typography>
                      <Flex gap='4'>
                        <div className='p-2 border rounded'>
                          <Typography variant='body-small'>Total</Typography>
                          <Typography variant='h4'>{currentAgentResponse.data.total_services}</Typography>
                        </div>
                        <div className='p-2 border rounded bg-palette-success-bg-weak'>
                          <Typography variant='body-small'>Healthy</Typography>
                          <Typography variant='h4' className='text-palette-success-text'>
                            {currentAgentResponse.data.healthy_services}
                          </Typography>
                        </div>
                        <div className='p-2 border rounded bg-palette-danger-bg-weak'>
                          <Typography variant='body-small'>Unhealthy</Typography>
                          <Typography variant='h4' className='text-palette-danger-text'>
                            {currentAgentResponse.data.unhealthy_services}
                          </Typography>
                        </div>
                      </Flex>
                    </div>
                  )}

                  {currentAgentResponse.data.services && currentAgentResponse.data.services.length > 0 && (
                    <div>
                      <Typography variant='h5' className='mb-2'>
                        Services Status
                      </Typography>
                      <div className='space-y-2'>
                        {currentAgentResponse.data.services.map((service, index) => (
                          <div key={index} className='p-2 border rounded flex justify-between'>
                            <Typography variant='body-medium'>{service.service}</Typography>
                            <Flex gap='2' alignItems='center'>
                              <StatusIndicator
                                type={
                                  service.status === 'Running'
                                    ? 'success'
                                    : service.status === 'Stopped'
                                      ? 'danger'
                                      : 'warning'
                                }
                              />
                              <Typography variant='body-small'>{service.status}</Typography>
                            </Flex>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {currentAgentResponse.data.messages && currentAgentResponse.data.messages.length > 0 && (
                    <div>
                      <Typography variant='h5' className='mb-2'>
                        Agent Messages
                      </Typography>
                      <div className='space-y-2 max-h-60 overflow-y-auto prose prose-sm'>
                        {currentAgentResponse.data.messages.map((msg, index) => (
                          <div key={index} className='p-2 border rounded bg-palette-neutral-bg-weak'>
                            <Typography variant='body-small' className='font-semibold'>
                              {msg.type}
                            </Typography>
                            <div className='mt-1'>
                              <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw] as any}>
                                {msg.content}
                              </ReactMarkdown>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              )}
            </Dialog.Content>
            <Dialog.Actions className='mt-4'>
              <Button onClick={handleCloseIssuesDialog}>Close</Button>
            </Dialog.Actions>
          </Dialog>
        </div>
        {isRightExpanded && (
          <DrawerChatbot
            messages={messages}
            isExpanded={isRightExpanded}
            clearHistoryData={clearHistoryData}
            connectionStatus={connectionStatus}
            setMessages={setMessages}
            isDeleteChatLoading={false}
          />
        )}
        <SideNav
          messages={messages}
          isExpanded={isRightExpanded}
          position='right'
          toggleDrawer={toggleRightDrawer}
          deleteOnClick={deleteOnClick}
          showDrawerChatbot={showDrawerChatbot}
          setShowDrawerChatbot={setShowDrawerChatbot}
          setIsRightExpanded={setIsRightExpanded}
          clearHistoryData={clearHistoryData}
          toggleGCSModal={() => {}}
          toggles3Modal={() => {}}
          toggleGenericModal={() => {}}
          setIsleftExpanded={setIsLeftExpanded}
        />
      </div>
    </>
  );
};

function useColumnHelper<T>() {
  return useMemo(() => createColumnHelper<T>(), []);
}

export default Dashboard;
